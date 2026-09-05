"""Held-out tasks: databases, migrations, and transactions.

These tasks target the failures that only surface at a boundary: a statement
that succeeded but should not have been kept, a row that was read correctly
and then written back over someone else's work, an event published for a
record that never committed. Code that passes a single-threaded happy path
routinely gets every one of them wrong.

Each validator drives a real SQLite connection opened with
``isolation_level=None``, so the candidate owns its own ``BEGIN``/``COMMIT``/
``ROLLBACK`` and transaction boundaries are observable rather than implied by
the driver. Several validators assert on ``connection.in_transaction`` after a
failure: an implementation that raises without unwinding leaves the connection
poisoned for every later caller, which is a defect the return value alone
cannot show.
"""

from __future__ import annotations

from scripts.programming_obstacle_tasks import task
from scripts.programming_obstacle_tasks._support import LOAD_CANDIDATE, require

FAMILY = "databases_migrations_transactions"

TASKS = [
    task(
        f"{FAMILY}-0001", FAMILY,
        prompt=(
            "Implement a Python function apply_migrations(connection, "
            "migrations) that brings a SQLite database up to date exactly "
            "once. `connection` is a sqlite3 connection opened with "
            "isolation_level=None, so your function issues its own "
            "transaction statements. `migrations` is a list of (version, "
            "name, sql) triples where version is an integer and sql is a "
            "single SQL statement. Create a schema_migrations table with "
            "columns (version INTEGER PRIMARY KEY, name TEXT NOT NULL, "
            "checksum TEXT NOT NULL) if it does not exist. Apply pending "
            "migrations in ascending version order; the checksum is the "
            "hex sha256 of the sql text encoded as UTF-8. A version already "
            "recorded is skipped, but if its stored checksum differs from "
            "the sql now supplied, raise ValueError -- an applied migration "
            "that was edited afterwards is a corrupted history, not a "
            "no-op. Raise ValueError for a pending migration whose version "
            "is not greater than the highest already-applied version. Apply "
            "each migration and record its row in one transaction, so a "
            "statement that fails leaves neither its effect nor its "
            "schema_migrations row behind, and the connection is left with "
            "no transaction open. Return the list of versions applied by "
            "this call, in the order applied."
        ),
        validator=LOAD_CANDIDATE + require("apply_migrations") + r'''
import hashlib
import sqlite3

def fresh():
    return sqlite3.connect(":memory:", isolation_level=None)

connection = fresh()
first = [
    (1, "create_users", "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT)"),
    (2, "create_posts", "CREATE TABLE posts (id INTEGER PRIMARY KEY, body TEXT)"),
]

# Migrations apply in ascending order, and report exactly what they applied.
assert apply_migrations(connection, first) == [1, 2]
tables = {
    row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )
}
assert {"users", "posts", "schema_migrations"} <= tables, tables

# The checksum stored is the sha256 of the sql, not of the name.
stored = dict(
    connection.execute("SELECT version, checksum FROM schema_migrations")
)
assert stored[1] == hashlib.sha256(first[0][2].encode("utf-8")).hexdigest()

# Re-running is a no-op: exactly-once is the whole contract.
assert apply_migrations(connection, first) == []
assert connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 2

# An applied migration whose text changed later is a corrupted history.
edited = list(first)
edited[0] = (1, "create_users", "CREATE TABLE users (id INTEGER PRIMARY KEY)")
try:
    apply_migrations(connection, edited)
except ValueError:
    pass
else:
    raise AssertionError("an edited applied migration must raise ValueError")

# A newly introduced version below the high-water mark cannot be inserted
# into history after the fact.
try:
    apply_migrations(connection, first + [(0, "backfill", "CREATE TABLE t0 (id INTEGER)")])
except ValueError:
    pass
else:
    raise AssertionError("an out-of-order pending migration must raise ValueError")

# A failing migration is atomic: no version row, and -- the part a bare
# `raise` gets wrong -- no transaction left open on the connection.
try:
    apply_migrations(connection, first + [(3, "bad", "INSERT INTO absent_table VALUES (1)")])
except ValueError:
    raise AssertionError("a broken statement is not an ordering error")
except Exception:
    pass
else:
    raise AssertionError("a failing migration must not report success")

assert connection.in_transaction is False, (
    "a failed migration must roll back, not leave the connection in a transaction"
)
recorded = {row[0] for row in connection.execute("SELECT version FROM schema_migrations")}
assert recorded == {1, 2}, recorded

# And the connection is still usable afterwards.
assert apply_migrations(connection, first + [(3, "later", "CREATE TABLE later (id INTEGER)")]) == [3]

# A fresh database applies everything from nothing.
assert apply_migrations(fresh(), first) == [1, 2]
'''),

    task(
        f"{FAMILY}-0002", FAMILY,
        prompt=(
            "Implement a Python function update_document(connection, doc_id, "
            "expected_version, fields) providing optimistic concurrency "
            "control over a table documents(id INTEGER PRIMARY KEY, title "
            "TEXT, body TEXT, version INTEGER NOT NULL). `fields` is a dict "
            "of column names to new values. Update the row only when its "
            "current version equals expected_version, setting the supplied "
            "columns and incrementing version by exactly one; return the new "
            "version. When the row is missing or its version has moved on, "
            "change nothing and return None -- a stale write is an ordinary "
            "outcome the caller retries, not an exception. Columns not "
            "present in `fields` keep their current values. Raise ValueError "
            "for an empty `fields` or any key that is not title or body; "
            "the column name is interpolated into SQL, so accepting an "
            "arbitrary key would be an injection point."
        ),
        validator=LOAD_CANDIDATE + require("update_document") + r'''
import sqlite3

connection = sqlite3.connect(":memory:", isolation_level=None)
connection.execute(
    "CREATE TABLE documents (id INTEGER PRIMARY KEY, title TEXT, body TEXT, "
    "version INTEGER NOT NULL)"
)
connection.execute(
    "INSERT INTO documents (id, title, body, version) VALUES (1, 'first', 'body', 1)"
)

def row():
    return connection.execute(
        "SELECT title, body, version FROM documents WHERE id = 1"
    ).fetchone()

# A matching version updates and bumps the version by exactly one.
assert update_document(connection, 1, 1, {"title": "second"}) == 2
assert row() == ("second", "body", 2)

# The losing writer of a concurrent pair sees None and overwrites nothing.
assert update_document(connection, 1, 1, {"title": "clobbered"}) is None
assert row() == ("second", "body", 2), "a stale write must not touch the row"

# Only the named columns move.
assert update_document(connection, 1, 2, {"body": "revised"}) == 3
assert row() == ("second", "revised", 3)

# Both columns at once still counts as one version.
assert update_document(connection, 1, 3, {"title": "third", "body": "again"}) == 4
assert row() == ("third", "again", 4)

# A missing row is a miss, not a crash.
assert update_document(connection, 99, 1, {"title": "ghost"}) is None

# The column allow-list is part of the contract, not a formality.
for bad in ({}, {"version": 9}, {"id": 2}, {"title = 'x' --": 1}):
    try:
        update_document(connection, 1, 4, bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{bad!r} must raise ValueError")

assert row() == ("third", "again", 4), "a rejected call must not have written"
'''),

    task(
        f"{FAMILY}-0003", FAMILY,
        prompt=(
            "Implement a Python function record_order(connection, order, "
            "event) writing a business row and its outbox event in one "
            "transaction. The schema is orders(id INTEGER PRIMARY KEY, "
            "customer TEXT NOT NULL, total_cents INTEGER NOT NULL) and "
            "outbox(id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT "
            "NULL, payload TEXT NOT NULL). `order` has keys id, customer "
            "and total_cents; `event` has keys topic and payload. Insert "
            "both and return the outbox row's id. The two writes must "
            "commit together or not at all: if the order cannot be inserted "
            "-- a duplicate id, for instance -- let the sqlite3 exception "
            "propagate, leave no outbox row behind, and leave no transaction "
            "open on the connection. An event describing an order that does "
            "not exist is the failure this pattern exists to prevent."
        ),
        validator=LOAD_CANDIDATE + require("record_order") + r'''
import sqlite3

connection = sqlite3.connect(":memory:", isolation_level=None)
connection.execute(
    "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer TEXT NOT NULL, "
    "total_cents INTEGER NOT NULL)"
)
connection.execute(
    "CREATE TABLE outbox (id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "topic TEXT NOT NULL, payload TEXT NOT NULL)"
)

def counts():
    return (
        connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0],
        connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0],
    )

event_id = record_order(
    connection,
    {"id": 1, "customer": "ada", "total_cents": 250},
    {"topic": "order.created", "payload": '{"id": 1}'},
)
assert isinstance(event_id, int)
assert counts() == (1, 1)
assert connection.execute(
    "SELECT topic, payload FROM outbox WHERE id = ?", (event_id,)
).fetchone() == ("order.created", '{"id": 1}')

# The duplicate order fails, and takes its event with it.
try:
    record_order(
        connection,
        {"id": 1, "customer": "grace", "total_cents": 900},
        {"topic": "order.created", "payload": '{"id": 1, "dup": true}'},
    )
except sqlite3.IntegrityError:
    pass
else:
    raise AssertionError("a duplicate order id must raise IntegrityError")

assert counts() == (1, 1), "a rolled-back order must not leave an outbox event"
assert connection.in_transaction is False, (
    "a failed write must roll back, not leave the connection in a transaction"
)

# The connection still works, and ids keep advancing.
second = record_order(
    connection,
    {"id": 2, "customer": "grace", "total_cents": 900},
    {"topic": "order.created", "payload": '{"id": 2}'},
)
assert second != event_id
assert counts() == (2, 2)
'''),

    task(
        f"{FAMILY}-0004", FAMILY,
        prompt=(
            "Implement a Python function page_after(connection, cursor, "
            "limit) doing keyset pagination over posts(id INTEGER PRIMARY "
            "KEY, created_at TEXT NOT NULL, title TEXT NOT NULL), ordered "
            "by created_at then id. `cursor` is None for the first page or "
            "the value returned by the previous call. Return a tuple "
            "(rows, next_cursor) where rows is a list of (id, created_at, "
            "title) tuples of at most `limit` entries in order, and "
            "next_cursor positions the following page. Return ([], None) "
            "when no rows remain. Timestamps are not unique, so the cursor "
            "must carry both sort keys and the comparison must be over the "
            "composite: a page boundary that lands in the middle of a group "
            "sharing one created_at value must not skip or repeat the rest "
            "of that group. Rows inserted before an already-returned "
            "position must not shift a later page."
        ),
        validator=LOAD_CANDIDATE + require("page_after") + r'''
import sqlite3

connection = sqlite3.connect(":memory:", isolation_level=None)
connection.execute(
    "CREATE TABLE posts (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, "
    "title TEXT NOT NULL)"
)
# Five of the seven rows share two timestamps, so every page boundary that
# matters falls inside a tie group.
seed = [
    (1, "2026-01-01", "a"), (2, "2026-01-01", "b"), (3, "2026-01-01", "c"),
    (4, "2026-01-02", "d"), (5, "2026-01-02", "e"),
    (6, "2026-01-03", "f"), (7, "2026-01-04", "g"),
]
connection.executemany("INSERT INTO posts VALUES (?, ?, ?)", seed)

def walk(limit):
    seen = []
    cursor = None
    for _ in range(50):
        rows, cursor = page_after(connection, cursor, limit)
        assert len(rows) <= limit, "a page must not exceed the limit"
        if not rows:
            assert cursor is None, "an empty page must not offer a next cursor"
            break
        seen.extend(tuple(row) for row in rows)
        if cursor is None:
            break
    else:
        raise AssertionError("pagination did not terminate")
    return seen

# Every page size must reconstruct the full ordering exactly once.
for limit in (1, 2, 3, 7, 10):
    walked = walk(limit)
    assert walked == seed, f"limit {limit} produced {walked}"
    assert len(walked) == len(set(row[0] for row in walked)), "a row was repeated"

# A boundary landing inside a tie group must resume inside that group.
rows, cursor = page_after(connection, None, 2)
assert [row[0] for row in rows] == [1, 2]
rows, cursor = page_after(connection, cursor, 2)
assert [row[0] for row in rows] == [3, 4], (
    "the cursor must compare the composite key, not created_at alone"
)

# Inserting behind the cursor must not disturb the rows still to come.
connection.execute("INSERT INTO posts VALUES (8, '2026-01-01', 'inserted')")
rows, cursor = page_after(connection, cursor, 10)
assert [row[0] for row in rows] == [5, 6, 7], [row[0] for row in rows]

rows, cursor = page_after(connection, cursor, 10)
assert rows == [] and cursor is None
'''),

    task(
        f"{FAMILY}-0005", FAMILY,
        prompt=(
            "Implement a Python function apply_batch(connection, items) that "
            "imports a batch where one bad row must not discard the good "
            "ones. The schema is accounts(id INTEGER PRIMARY KEY, owner TEXT "
            "NOT NULL) and audit(account_id INTEGER NOT NULL, amount INTEGER "
            "NOT NULL CHECK (amount >= 0)). Each item is a dict with keys "
            "id, owner and amount. For each item, in order, insert the "
            "accounts row and then the audit row. Wrap the whole batch in "
            "one transaction, and each item in a SQLite SAVEPOINT so a "
            "failing item is rolled back to its own savepoint -- including "
            "the accounts row it had already written before the audit row "
            "was rejected -- while earlier and later items still commit. "
            "Return a tuple (applied, rejected): applied is the list of ids "
            "that committed, rejected the list of (id, exception class name) "
            "pairs, both in input order."
        ),
        validator=LOAD_CANDIDATE + require("apply_batch") + r'''
import sqlite3

connection = sqlite3.connect(":memory:", isolation_level=None)
connection.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, owner TEXT NOT NULL)")
connection.execute(
    "CREATE TABLE audit (account_id INTEGER NOT NULL, "
    "amount INTEGER NOT NULL CHECK (amount >= 0))"
)

items = [
    {"id": 1, "owner": "ada", "amount": 10},
    {"id": 2, "owner": "bob", "amount": -5},    # audit CHECK rejects this one
    {"id": 3, "owner": "cyd", "amount": 30},
    {"id": 1, "owner": "dup", "amount": 40},    # duplicate primary key
]
applied, rejected = apply_batch(connection, items)

assert applied == [1, 3], applied
assert [entry[0] for entry in rejected] == [2, 1], rejected
assert all(isinstance(entry[1], str) for entry in rejected), rejected

# The partial write of item 2 -- its accounts row -- must be gone. Releasing
# the savepoint instead of rolling back to it leaves account 2 behind.
stored = [row[0] for row in connection.execute("SELECT id FROM accounts ORDER BY id")]
assert stored == [1, 3], stored
assert connection.execute(
    "SELECT owner FROM accounts WHERE id = 1"
).fetchone()[0] == "ada", "the duplicate must not have overwritten the original"

audited = [tuple(row) for row in connection.execute(
    "SELECT account_id, amount FROM audit ORDER BY account_id"
)]
assert audited == [(1, 10), (3, 30)], audited

# The batch committed, and left nothing open.
assert connection.in_transaction is False
assert apply_batch(connection, []) == ([], [])
'''),

    task(
        f"{FAMILY}-0006", FAMILY,
        prompt=(
            "Implement a Python function merge_record(connection, record) "
            "performing a partial upsert into profiles(id INTEGER PRIMARY "
            "KEY, display_name TEXT, email TEXT, locale TEXT). `record` is a "
            "dict always containing id and any subset of the other three "
            "columns. A column whose value is None, and a column absent from "
            "the dict, both mean leave the stored value alone -- None is "
            "'not supplied', never 'set to NULL'. Insert the row when the id "
            "is absent, writing NULL for columns that were not supplied. "
            "Return the string 'inserted' or 'updated' accordingly. Raise "
            "ValueError when id is missing or when the dict carries a key "
            "that is not one of the four columns."
        ),
        validator=LOAD_CANDIDATE + require("merge_record") + r'''
import sqlite3

connection = sqlite3.connect(":memory:", isolation_level=None)
connection.execute(
    "CREATE TABLE profiles (id INTEGER PRIMARY KEY, display_name TEXT, "
    "email TEXT, locale TEXT)"
)

def row(row_id=1):
    return connection.execute(
        "SELECT display_name, email, locale FROM profiles WHERE id = ?", (row_id,)
    ).fetchone()

assert merge_record(connection, {
    "id": 1, "display_name": "Ada", "email": "ada@example.com", "locale": "en"
}) == "inserted"
assert row() == ("Ada", "ada@example.com", "en")

# An absent key leaves the column alone.
assert merge_record(connection, {"id": 1, "email": "ada@lovelace.example"}) == "updated"
assert row() == ("Ada", "ada@lovelace.example", "en")

# An explicit None means 'not supplied' -- this is the case that silently
# erases a user's data when None is passed straight through to the UPDATE.
assert merge_record(connection, {
    "id": 1, "display_name": None, "email": None, "locale": "fr"
}) == "updated"
assert row() == ("Ada", "ada@lovelace.example", "fr"), (
    "None must preserve the stored value, not overwrite it with NULL"
)

# An update naming no columns at all is still an update, and changes nothing.
assert merge_record(connection, {"id": 1}) == "updated"
assert row() == ("Ada", "ada@lovelace.example", "fr")

# Insert with a partial record leaves the rest NULL.
assert merge_record(connection, {"id": 2, "email": "grace@example.com"}) == "inserted"
assert row(2) == (None, "grace@example.com", None)

# Insert where a supplied value is None behaves the same as omitting it.
assert merge_record(connection, {"id": 3, "display_name": None, "locale": "de"}) == "inserted"
assert row(3) == (None, None, "de")

for bad in ({}, {"display_name": "x"}, {"id": 1, "role": "admin"}):
    try:
        merge_record(connection, bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{bad!r} must raise ValueError")

assert connection.execute("SELECT COUNT(*) FROM profiles").fetchone()[0] == 3
'''),

    task(
        f"{FAMILY}-0007", FAMILY,
        prompt=(
            "Implement a Python function delete_customer(connection, "
            "customer_id) that refuses to orphan rows. The schema is "
            "customers(id INTEGER PRIMARY KEY, name TEXT NOT NULL) and "
            "orders(id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL "
            "REFERENCES customers(id)). Delete the customer and return True "
            "when nothing references it. When orders still reference the "
            "customer, change nothing and return False. Return False for a "
            "customer id that does not exist. SQLite does not enforce "
            "foreign keys unless they are switched on for the connection, "
            "and the pragma that does so is a silent no-op inside an open "
            "transaction -- so the deletion must actually be prevented, not "
            "merely intended. Leave no transaction open on return."
        ),
        validator=LOAD_CANDIDATE + require("delete_customer") + r'''
import sqlite3

connection = sqlite3.connect(":memory:", isolation_level=None)
connection.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
connection.execute(
    "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL "
    "REFERENCES customers(id))"
)
connection.executemany(
    "INSERT INTO customers VALUES (?, ?)", [(1, "ada"), (2, "grace"), (3, "lonely")]
)
connection.executemany("INSERT INTO orders VALUES (?, ?)", [(10, 1), (11, 1), (12, 2)])

def customers():
    return [row[0] for row in connection.execute("SELECT id FROM customers ORDER BY id")]

# A referenced customer survives, and so do the orders.
assert delete_customer(connection, 1) is False
assert customers() == [1, 2, 3], "a referenced customer must not be deleted"
assert connection.execute(
    "SELECT COUNT(*) FROM orders WHERE customer_id = 1"
).fetchone()[0] == 2, "the orders must be untouched"
assert connection.in_transaction is False

# An unreferenced customer goes.
assert delete_customer(connection, 3) is True
assert customers() == [1, 2]
assert connection.in_transaction is False

# A missing id is False, not an exception.
assert delete_customer(connection, 999) is False
assert customers() == [1, 2]

# Once the last order is gone the customer can be removed -- proving the
# refusal was a live constraint check, not a blanket refusal.
connection.execute("DELETE FROM orders WHERE customer_id = 2")
assert delete_customer(connection, 2) is True
assert customers() == [1]
'''),

    task(
        f"{FAMILY}-0008", FAMILY,
        prompt=(
            "Implement a Python function process_payment(connection, "
            "idempotency_key, amount_cents) charging at most once per key. "
            "The schema is payments(charge_id INTEGER PRIMARY KEY "
            "AUTOINCREMENT, idempotency_key TEXT NOT NULL UNIQUE, "
            "amount_cents INTEGER NOT NULL). On a key never seen, insert the "
            "row and return {'charge_id': <new id>, 'amount_cents': "
            "<amount>, 'replayed': False}. On a key already stored, insert "
            "nothing and return the stored charge_id and the stored "
            "amount_cents with 'replayed': True -- a retried request must "
            "return the original outcome even when it arrives with a "
            "different amount, because the caller is retrying, not "
            "re-pricing. Raise ValueError for an amount that is not a "
            "positive integer, before writing anything; a bool is not an "
            "acceptable amount even though it is an int. Leave no "
            "transaction open on return."
        ),
        validator=LOAD_CANDIDATE + require("process_payment") + r'''
import sqlite3

connection = sqlite3.connect(":memory:", isolation_level=None)
connection.execute(
    "CREATE TABLE payments (charge_id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "idempotency_key TEXT NOT NULL UNIQUE, amount_cents INTEGER NOT NULL)"
)

def charges():
    return connection.execute("SELECT COUNT(*) FROM payments").fetchone()[0]

first = process_payment(connection, "key-a", 500)
assert first["replayed"] is False
assert first["amount_cents"] == 500
assert isinstance(first["charge_id"], int)
assert charges() == 1

# The retry returns the original charge and does not bill again.
replay = process_payment(connection, "key-a", 500)
assert replay["replayed"] is True
assert replay["charge_id"] == first["charge_id"]
assert replay["amount_cents"] == 500
assert charges() == 1, "a replayed key must not create a second charge"

# A retry carrying a different amount still returns the stored one.
drifted = process_payment(connection, "key-a", 9999)
assert drifted["replayed"] is True
assert drifted["charge_id"] == first["charge_id"]
assert drifted["amount_cents"] == 500, (
    "the stored amount wins: a retry is not a re-price"
)
assert charges() == 1

# A different key is a different charge.
second = process_payment(connection, "key-b", 750)
assert second["replayed"] is False
assert second["charge_id"] != first["charge_id"]
assert second["amount_cents"] == 750
assert charges() == 2

# Validation happens before any write.
for bad in (0, -1, 2.5, "500", None, True):
    try:
        process_payment(connection, "key-c", bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"amount {bad!r} must raise ValueError")

assert charges() == 2, "a rejected amount must not have been written"
assert connection.in_transaction is False
'''),

    task(
        f"{FAMILY}-0009", FAMILY,
        prompt=(
            "Implement a Python function build_filter_query(table, filters, "
            "allowed_columns) returning a (sql, params) pair for a "
            "parameterised SELECT. `allowed_columns` is the set of columns a "
            "caller may filter on. Each entry in `filters` maps a column to "
            "either a plain value, meaning equality, or a two-element list "
            "or tuple (operator, value); a list or tuple value is always an "
            "operator pair, never a plain value. Supported operators are "
            "'=', '!=', '<', '<=', '>', '>=' and 'IN'. Emit "
            "\"SELECT * FROM <table>\" followed, when there is at least one "
            "filter, by \" WHERE \" and the conditions joined with \" AND \" "
            "in ascending column-name order, so the same filters always "
            "produce the same SQL. Every value must travel in `params` "
            "behind a '?' placeholder and must never be interpolated into "
            "the SQL text. 'IN' requires a list or tuple and expands to one "
            "placeholder per element as \"col IN (?, ?)\"; an empty sequence "
            "matches nothing and renders as the condition \"1 = 0\" with no "
            "placeholder. Raise ValueError when the table name is not a bare "
            "identifier of ASCII letters, digits and underscores not "
            "starting with a digit, when a filter column is not in "
            "allowed_columns, when an operator pair is not exactly two "
            "elements, when the operator is unsupported, or when 'IN' is "
            "given something other than a list or tuple."
        ),
        validator=LOAD_CANDIDATE + require("build_filter_query") + r'''
import sqlite3

db = sqlite3.connect(":memory:")
db.execute(
    "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)"
)
db.executemany("INSERT INTO users VALUES (?, ?, ?)", [
    (1, "ada", 36), (2, "grace", 45), (3, "alan", 41), (4, "katherine", 52),
])
db.commit()
ALLOWED = {"id", "name", "age"}


def run(filters):
    sql, params = build_filter_query("users", filters, ALLOWED)
    return sorted(row[0] for row in db.execute(sql, params))


# The generated SQL actually selects the right rows.
assert run({}) == [1, 2, 3, 4]
assert run({"name": "ada"}) == [1]
assert run({"age": (">=", 45)}) == [2, 4]
assert run({"age": ("<", 41)}) == [1]
assert run({"age": ("<=", 41)}) == [1, 3]
assert run({"age": (">", 41), "name": ("!=", "grace")}) == [4]
assert run({"id": ("IN", [1, 3])}) == [1, 3]
assert run({"id": ("IN", (2,))}) == [2]
assert run({"id": ("IN", [])}) == []
assert run({"name": "nobody"}) == []

# No filters means no WHERE clause and no parameters.
sql, params = build_filter_query("users", {}, ALLOWED)
assert sql == "SELECT * FROM users", sql
assert list(params) == []

# The rendering is deterministic and ordered by column name.
sql, params = build_filter_query("users", {"name": "ada", "age": 36}, ALLOWED)
assert sql == "SELECT * FROM users WHERE age = ? AND name = ?", sql
assert list(params) == [36, "ada"]

sql, params = build_filter_query("users", {"id": ("IN", [1, 2])}, ALLOWED)
assert sql == "SELECT * FROM users WHERE id IN (?, ?)", sql
assert list(params) == [1, 2]

sql, params = build_filter_query("users", {"id": ("IN", [])}, ALLOWED)
assert sql == "SELECT * FROM users WHERE 1 = 0", sql
assert list(params) == []

# The value travels as data. This is the whole point of the exercise: the
# hostile string must appear in params and never in the SQL.
hostile = "ada'; DROP TABLE users; --"
sql, params = build_filter_query("users", {"name": hostile}, ALLOWED)
assert hostile not in sql, "the value was interpolated into the SQL text"
assert list(params) == [hostile]
assert list(db.execute(sql, params)) == []
assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 4, (
    "the table did not survive executing the generated query"
)

# A NULL and a zero are ordinary values, not absent ones.
sql, params = build_filter_query("users", {"age": None}, ALLOWED)
assert list(params) == [None]
sql, params = build_filter_query("users", {"age": 0}, ALLOWED)
assert list(params) == [0]


def rejects(table, filters, why):
    try:
        build_filter_query(table, filters, ALLOWED)
    except ValueError:
        return
    raise AssertionError(why)


# An identifier cannot be parameterised, so it must be allow-listed.
rejects("users", {"password_hash": "x"}, "a column outside the allow-list was accepted")
rejects("users", {"name; DROP TABLE users": "x"}, "an injected column was accepted")
rejects("users", {"1name": "x"}, "an invalid column identifier was accepted")
rejects("users; DROP TABLE users", {"name": "ada"}, "an injected table was accepted")
rejects("", {"name": "ada"}, "an empty table name was accepted")
rejects("1users", {"name": "ada"}, "a table name starting with a digit was accepted")
rejects(None, {"name": "ada"}, "a non-string table name was accepted")

# Malformed conditions.
rejects("users", {"name": ("LIKE", "a%")}, "an unsupported operator was accepted")
rejects("users", {"name": ("=",)}, "a one-element operator pair was accepted")
rejects("users", {"name": ("=", "a", "b")}, "a three-element operator pair was accepted")
rejects("users", {"name": ()}, "an empty operator pair was accepted")
rejects("users", {"id": ("IN", "ada")}, "IN with a bare string was accepted")
rejects("users", {"id": ("IN", 7)}, "IN with a non-sequence was accepted")
'''),

    task(
        f"{FAMILY}-0010", FAMILY,
        prompt=(
            "Implement a Python function apply_line_items(connection, "
            "order_id, items). `connection` is a sqlite3 connection opened "
            "with isolation_level=None, so your function issues its own "
            "transaction statements. The schema already holds line_items "
            "(order_id, sku, quantity) with quantity constrained positive "
            "and (order_id, sku) unique, and order_totals(order_id PRIMARY "
            "KEY, total) with total constrained to at most 100. `items` is a "
            "list of (sku, quantity) pairs. Ensure the order's totals row "
            "exists starting at zero, then apply each item in turn: insert "
            "the line item and add its quantity to the running total. "
            "Applying an item takes two statements and must be all or "
            "nothing -- an item rejected because its sku repeats, its "
            "quantity is not positive, or it would push the total past the "
            "cap must leave neither the line-items row nor any change to the "
            "total, and the batch must carry on with the remaining items. "
            "The call as a whole is one transaction that commits at the end, "
            "so a concurrent reader sees either the state before the call or "
            "every accepted item; on an unexpected error roll the whole "
            "thing back and re-raise. Leave no transaction open. Return the "
            "ascending list of indices into `items` that were rejected."
        ),
        validator=LOAD_CANDIDATE + require("apply_line_items") + r'''
import sqlite3


def fresh():
    db = sqlite3.connect(":memory:", isolation_level=None)
    db.execute(
        "CREATE TABLE line_items ("
        " order_id INTEGER NOT NULL, sku TEXT NOT NULL,"
        " quantity INTEGER NOT NULL CHECK (quantity > 0),"
        " UNIQUE (order_id, sku))"
    )
    db.execute(
        "CREATE TABLE order_totals ("
        " order_id INTEGER PRIMARY KEY,"
        " total INTEGER NOT NULL CHECK (total <= 100))"
    )
    return db


def rows(db, order_id=1):
    return sorted(db.execute(
        "SELECT sku, quantity FROM line_items WHERE order_id = ?", (order_id,)
    ))


def total(db, order_id=1):
    row = db.execute(
        "SELECT total FROM order_totals WHERE order_id = ?", (order_id,)
    ).fetchone()
    return None if row is None else row[0]


# An item rejected by the SECOND of its two statements must undo the first.
db = fresh()
assert apply_line_items(db, 1, [("a", 60), ("b", 60), ("c", 30)]) == [1]
assert total(db) == 90, total(db)
assert rows(db) == [("a", 60), ("c", 30)], rows(db)
assert db.execute(
    "SELECT COUNT(*) FROM line_items WHERE sku = 'b'"
).fetchone()[0] == 0, (
    "the rejected item left its line_items row behind, so the insert was "
    "not covered by the same unit of work as the total update"
)
assert db.in_transaction is False, "the connection was left in a transaction"

# A duplicate sku within the batch is rejected without losing the first.
db = fresh()
assert apply_line_items(db, 1, [("a", 1), ("a", 2), ("b", 3)]) == [1]
assert rows(db) == [("a", 1), ("b", 3)], rows(db)
assert total(db) == 4, total(db)

# A non-positive quantity is rejected by the insert itself.
db = fresh()
assert apply_line_items(db, 1, [("a", 0), ("b", -1), ("c", 5)]) == [0, 1]
assert rows(db) == [("c", 5)]
assert total(db) == 5

# Every item rejected still leaves a committed, consistent state.
db = fresh()
assert apply_line_items(db, 1, [("a", 0), ("b", -1)]) == [0, 1]
assert rows(db) == [] and total(db) == 0
assert db.in_transaction is False

# Nothing rejected.
db = fresh()
assert apply_line_items(db, 1, [("a", 1), ("b", 2)]) == []
assert rows(db) == [("a", 1), ("b", 2)] and total(db) == 3

# An empty batch is valid and still leaves the totals row consistent.
db = fresh()
assert apply_line_items(db, 1, []) == []
assert total(db) == 0
assert db.in_transaction is False

# The cap is per order, so two orders may repeat a sku and each fill up.
db = fresh()
assert apply_line_items(db, 1, [("a", 100)]) == []
assert apply_line_items(db, 2, [("a", 100)]) == []
assert total(db, 1) == 100 and total(db, 2) == 100

# Exactly at the cap is accepted; one past it is not.
db = fresh()
assert apply_line_items(db, 1, [("a", 100), ("b", 1)]) == [1]
assert total(db) == 100 and rows(db) == [("a", 100)]

# Accepted work is committed, not merely visible on this connection: a
# later transaction that rolls back cannot take it away.
db = fresh()
apply_line_items(db, 3, [("z", 4)])
db.execute("BEGIN")
db.execute("ROLLBACK")
assert db.execute(
    "SELECT COUNT(*) FROM line_items WHERE order_id = 3"
).fetchone()[0] == 1, "the accepted rows were never committed"

# A second call adds to the existing total rather than resetting it.
db = fresh()
assert apply_line_items(db, 1, [("a", 40)]) == []
assert apply_line_items(db, 1, [("b", 40)]) == []
assert total(db) == 80
'''),

    task(
        f"{FAMILY}-0011", FAMILY,
        prompt=(
            "Implement a Python function summarize_regions(connection) "
            "reporting sales per region from a table sales(region TEXT NOT "
            "NULL, amount INTEGER) where amount may be NULL, meaning the "
            "sale was recorded but its value has not been reported yet. "
            "Return a dict mapping each region that has at least one row to "
            "a dict with four keys: 'rows', the number of rows for that "
            "region; 'reported', the number of those rows whose amount is "
            "not NULL; 'total', the sum of the non-NULL amounts, which is 0 "
            "rather than None when the region has no reported amounts; and "
            "'average', the mean of the non-NULL amounts as a float, or None "
            "when there are none, since a mean over nothing is undefined "
            "rather than zero. A region whose rows are all unreported must "
            "still appear, with rows counted and reported zero."
        ),
        validator=LOAD_CANDIDATE + require("summarize_regions") + r'''
import sqlite3

db = sqlite3.connect(":memory:")
db.execute("CREATE TABLE sales (region TEXT NOT NULL, amount INTEGER)")
db.executemany("INSERT INTO sales VALUES (?, ?)", [
    ("north", 100), ("north", 200), ("north", None),
    ("south", 50),
    ("east", None), ("east", None),
    ("west", 0),
])
db.commit()

summary = summarize_regions(db)
assert set(summary) == {"north", "south", "east", "west"}, sorted(summary)

# COUNT(*) and COUNT(amount) are different questions.
assert summary["north"]["rows"] == 3, summary["north"]
assert summary["north"]["reported"] == 2, summary["north"]
assert summary["north"]["total"] == 300
assert abs(summary["north"]["average"] - 150.0) < 1e-9

assert summary["south"] == {
    "rows": 1, "reported": 1, "total": 50, "average": 50.0,
}, summary["south"]

# A region with nothing reported still exists, and SUM over no rows is NULL
# in SQL -- which must surface as 0, while the mean stays undefined.
assert summary["east"]["rows"] == 2, summary["east"]
assert summary["east"]["reported"] == 0, summary["east"]
assert summary["east"]["total"] == 0, (
    "SUM over no non-NULL amounts is NULL in SQL and must be reported as 0"
)
assert summary["east"]["average"] is None, (
    "a mean over zero reported amounts is undefined, not zero"
)

# A reported zero is a reported amount, not a missing one.
assert summary["west"] == {
    "rows": 1, "reported": 1, "total": 0, "average": 0.0,
}, summary["west"]

# An empty table yields an empty summary rather than a row of zeros.
empty = sqlite3.connect(":memory:")
empty.execute("CREATE TABLE sales (region TEXT NOT NULL, amount INTEGER)")
assert summarize_regions(empty) == {}

# Negative amounts are ordinary and must not be confused with missing ones.
signed = sqlite3.connect(":memory:")
signed.execute("CREATE TABLE sales (region TEXT NOT NULL, amount INTEGER)")
signed.executemany("INSERT INTO sales VALUES (?, ?)", [
    ("refunds", -10), ("refunds", 10), ("refunds", None),
])
signed.commit()
result = summarize_regions(signed)["refunds"]
assert result["rows"] == 3 and result["reported"] == 2
assert result["total"] == 0 and abs(result["average"] - 0.0) < 1e-9

# The average is a float even when the division is exact.
assert isinstance(summary["south"]["average"], float)
'''),

    task(
        f"{FAMILY}-0012", FAMILY,
        prompt=(
            "Implement a Python function lapsed_customers(connection, since) "
            "returning the sorted list of names of customers who have placed "
            "no order on or after `since`. The schema is customers(id "
            "INTEGER PRIMARY KEY, name TEXT NOT NULL) and orders(id INTEGER "
            "PRIMARY KEY, customer_id INTEGER NOT NULL, placed_on TEXT NOT "
            "NULL), where placed_on is an ISO date that sorts correctly as "
            "text. A customer with no orders at all has placed no recent "
            "order and must be included. A customer with both an old order "
            "and a recent one has placed a recent order and must be "
            "excluded. An order placed exactly on `since` counts as recent. "
            "Note that filtering the joined order rows in a WHERE clause "
            "answers a different question -- it finds customers who have an "
            "old order, which is neither necessary nor sufficient for having "
            "no recent one."
        ),
        validator=LOAD_CANDIDATE + require("lapsed_customers") + r'''
import sqlite3

db = sqlite3.connect(":memory:")
db.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
db.execute(
    "CREATE TABLE orders (id INTEGER PRIMARY KEY,"
    " customer_id INTEGER NOT NULL, placed_on TEXT NOT NULL)"
)
db.executemany("INSERT INTO customers VALUES (?, ?)", [
    (1, "alice"), (2, "bob"), (3, "cara"), (4, "dan"), (5, "eve"),
])
db.executemany("INSERT INTO orders VALUES (?, ?, ?)", [
    # alice has no orders at all.
    (10, 2, "2026-01-01"),                      # bob: only an old order
    (11, 3, "2026-08-01"),                      # cara: only a recent order
    (12, 4, "2026-01-15"), (13, 4, "2026-08-02"),  # dan: both
    (14, 5, "2026-06-01"),                      # eve: exactly on the boundary
])
db.commit()

SINCE = "2026-06-01"

result = lapsed_customers(db, SINCE)
assert result == ["alice", "bob"], result

# Each of the four ways to get this wrong, named:
assert "alice" in result, (
    "a customer with no orders was dropped, which is what happens when the "
    "date filter sits in WHERE and discards the NULL row of a LEFT JOIN"
)
assert "dan" not in result, (
    "a customer with an old order AND a recent one was reported as lapsed"
)
assert "cara" not in result, "a customer with only a recent order was reported"
assert "eve" not in result, (
    "an order placed exactly on `since` must count as recent"
)

# The result is sorted and free of duplicates even though dan has two orders.
assert result == sorted(result)
assert len(result) == len(set(result))

# Moving the boundary past every order lapses everyone who exists.
assert lapsed_customers(db, "2027-01-01") == [
    "alice", "bob", "cara", "dan", "eve",
]

# Moving it before every order lapses only the customer with no orders.
assert lapsed_customers(db, "2020-01-01") == ["alice"]

# A database with no orders at all lapses every customer.
bare = sqlite3.connect(":memory:")
bare.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
bare.execute(
    "CREATE TABLE orders (id INTEGER PRIMARY KEY,"
    " customer_id INTEGER NOT NULL, placed_on TEXT NOT NULL)"
)
bare.execute("INSERT INTO customers VALUES (1, 'solo')")
bare.commit()
assert lapsed_customers(bare, SINCE) == ["solo"]

# No customers at all is an empty list, not an error.
bare.execute("DELETE FROM customers")
assert lapsed_customers(bare, SINCE) == []

# An order belonging to no customer must not resurrect anyone.
bare.execute("INSERT INTO customers VALUES (1, 'solo')")
bare.execute("INSERT INTO orders VALUES (99, 42, '2026-09-01')")
bare.commit()
assert lapsed_customers(bare, SINCE) == ["solo"]
'''),
]
