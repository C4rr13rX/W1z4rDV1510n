"""Held-out tasks: HTTP APIs, authentication, and application security.

Security defects are the case where a passing functional test proves least.
Every task here has an implementation that satisfies the happy path and is
still exploitable, so each validator drives the attack rather than the feature:
a token whose header was swapped after signing, a path that normalises to a
sibling of the root, a redirect that is protocol-relative rather than absolute,
a cookie value carrying CRLF, a token minted for one session presented with
another.

Validators are deterministic -- clocks and salts are injected or observed
through behaviour, never sampled from the wall clock -- because a security
check that is only usually right is the defect, not the test.
"""

from __future__ import annotations

from scripts.programming_obstacle_tasks import task
from scripts.programming_obstacle_tasks._support import LOAD_CANDIDATE, require

FAMILY = "http_apis_authn_appsec"

TASKS = [
    task(
        f"{FAMILY}-0001", FAMILY,
        prompt=(
            "Implement a Python function verify_token(token, key, now) "
            "validating a compact HMAC-signed token of the form "
            "base64url(header).base64url(payload).base64url(signature), "
            "where base64url segments carry no '=' padding. `key` is bytes or "
            "str, `now` an integer POSIX timestamp. Decode the JSON header "
            "and payload. Accept only the exact algorithm 'HS256'; any other "
            "value, including 'none', is a failure. The signature covers the "
            "ASCII bytes of the first two segments joined by a '.', so a "
            "header altered after signing must not verify. Compare signatures "
            "with hmac.compare_digest. Require an integer 'exp' claim in the "
            "payload and treat the token as expired when now is greater than "
            "or equal to it. Return the payload dict when every check passes; "
            "raise ValueError otherwise. Never return a payload you have not "
            "verified."
        ),
        validator=LOAD_CANDIDATE + require("verify_token") + r'''
import base64
import hashlib
import hmac
import json

KEY = b"correct-horse-battery-staple"

def b64(raw):
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

def mint(header, payload, key=KEY):
    head = b64(json.dumps(header, sort_keys=True).encode("utf-8"))
    body = b64(json.dumps(payload, sort_keys=True).encode("utf-8"))
    signature = hmac.new(key, f"{head}.{body}".encode("ascii"), hashlib.sha256).digest()
    return f"{head}.{body}.{b64(signature)}"

good = mint({"alg": "HS256", "typ": "JWT"}, {"sub": "ada", "exp": 2000})
assert verify_token(good, KEY, 1999) == {"sub": "ada", "exp": 2000}

# A str key must work the same as bytes.
assert verify_token(good, KEY.decode("ascii"), 1999)["sub"] == "ada"

def rejects(token, key, now, why):
    try:
        verify_token(token, key, now)
    except ValueError:
        return
    raise AssertionError(why)

# Expiry is inclusive at the boundary.
rejects(good, KEY, 2000, "a token must be expired at exactly exp")
rejects(good, KEY, 2001, "an expired token must be rejected")

# The signature is over header AND payload, so re-heading a signed token
# must not verify -- this is the alg-confusion family.
head, body, signature = good.split(".")
reheaded = b64(json.dumps({"alg": "HS256", "kid": "evil"}, sort_keys=True).encode("utf-8"))
rejects(f"{reheaded}.{body}.{signature}", KEY, 1999,
        "a header swapped after signing must not verify")

# alg: none, with and without a signature.
none_head = b64(json.dumps({"alg": "none"}, sort_keys=True).encode("utf-8"))
rejects(f"{none_head}.{body}.", KEY, 1999, "alg none must be rejected")
rejects(f"{none_head}.{body}.{signature}", KEY, 1999, "alg none must be rejected")
rejects(mint({"alg": "HS512"}, {"sub": "ada", "exp": 2000}), KEY, 1999,
        "an unexpected algorithm must be rejected")

# Wrong key, tampered payload, structural damage.
rejects(good, b"wrong-key", 1999, "a bad signature must be rejected")
tampered = b64(json.dumps({"sub": "root", "exp": 2000}, sort_keys=True).encode("utf-8"))
rejects(f"{head}.{tampered}.{signature}", KEY, 1999, "a tampered payload must be rejected")
for broken in ("", "a.b", "a.b.c.d", "!!!.???.***", f"{head}.{body}"):
    rejects(broken, KEY, 1999, f"{broken!r} must be rejected")

# A missing or non-integer exp is a failure, not an eternal token.
rejects(mint({"alg": "HS256"}, {"sub": "ada"}), KEY, 1999, "a missing exp must be rejected")
rejects(mint({"alg": "HS256"}, {"sub": "ada", "exp": "2000"}), KEY, 1999,
        "a string exp must be rejected")
'''),

    task(
        f"{FAMILY}-0002", FAMILY,
        prompt=(
            "Implement a Python function resolve_within_root(root, "
            "user_path) that maps an untrusted relative path to a location "
            "guaranteed to sit inside `root`. Both are POSIX-style strings; "
            "do not touch the filesystem. Normalise the join of root and "
            "user_path and return the normalised absolute result. Raise "
            "ValueError when user_path is empty or not a string, is "
            "absolute, contains a NUL byte, or normalises to anything "
            "outside root. Containment is on a path-segment boundary: for "
            "root '/srv/data', the path '/srv/data-private/secrets' is "
            "outside even though its text starts with the root's text. The "
            "root itself is a valid result."
        ),
        validator=LOAD_CANDIDATE + require("resolve_within_root") + r'''
ROOT = "/srv/data"

assert resolve_within_root(ROOT, "notes.txt") == "/srv/data/notes.txt"
assert resolve_within_root(ROOT, "a/b/c.txt") == "/srv/data/a/b/c.txt"
assert resolve_within_root(ROOT, "./notes.txt") == "/srv/data/notes.txt"
assert resolve_within_root(ROOT, "a/../notes.txt") == "/srv/data/notes.txt"
assert resolve_within_root(ROOT, "a//b.txt") == "/srv/data/a/b.txt"
assert resolve_within_root(ROOT, ".") == "/srv/data"
assert resolve_within_root(ROOT, "a/b/../..") == "/srv/data"

# A trailing slash on the root must not change the verdict.
assert resolve_within_root("/srv/data/", "notes.txt") == "/srv/data/notes.txt"

def rejects(path, why, root=ROOT):
    try:
        resolve_within_root(root, path)
    except ValueError:
        return
    raise AssertionError(why)

rejects("../etc/passwd", "a traversal must be rejected")
rejects("a/../../etc/passwd", "a traversal through a subdirectory must be rejected")
rejects("/etc/passwd", "an absolute path must be rejected")
rejects("..", "the parent of the root is outside it")
rejects("", "an empty path must be rejected")
rejects("a/\x00b", "a NUL byte must be rejected")

# The boundary case: a sibling whose name extends the root's text.
rejects("../data-private/secrets",
        "containment must be on a segment boundary, not a text prefix")

for bad in (None, 42, b"notes.txt", ["notes.txt"]):
    try:
        resolve_within_root(ROOT, bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{bad!r} must raise ValueError")
'''),

    task(
        f"{FAMILY}-0003", FAMILY,
        prompt=(
            "Implement two Python functions, hash_password(password) and "
            "verify_password(password, stored). hash_password derives a key "
            "with hashlib.pbkdf2_hmac('sha256', ...) using a fresh "
            "cryptographically random salt of at least 16 bytes and at least "
            "100000 iterations, returning the string "
            "'pbkdf2_sha256$<iterations>$<b64 salt>$<b64 derived key>' with "
            "standard base64. verify_password parses a stored string, "
            "re-derives with the parameters recorded in it -- not with "
            "today's defaults, so old hashes keep verifying after the cost is "
            "raised -- and returns True or False, comparing with "
            "hmac.compare_digest. A malformed or unknown-scheme stored value "
            "returns False rather than raising. Passwords are str and are "
            "encoded as UTF-8. Two calls to hash_password with the same "
            "password must return different strings."
        ),
        validator=LOAD_CANDIDATE + require("hash_password") + require("verify_password") + r'''
import base64

stored = hash_password("hunter2")
assert isinstance(stored, str)
scheme, iterations, salt_b64, key_b64 = stored.split("$")
assert scheme == "pbkdf2_sha256", scheme
assert int(iterations) >= 100000, iterations
assert len(base64.b64decode(salt_b64)) >= 16, "the salt must be at least 16 bytes"
assert len(base64.b64decode(key_b64)) >= 16

assert verify_password("hunter2", stored) is True
assert verify_password("Hunter2", stored) is False
assert verify_password("", stored) is False
assert verify_password("hunter2 ", stored) is False

# A per-password salt: identical passwords must not produce identical hashes,
# or one rainbow table covers every user who chose the same password.
again = hash_password("hunter2")
assert again != stored, "each hash must use a fresh random salt"
assert again.split("$")[2] != salt_b64, "the salt must differ between hashes"
assert verify_password("hunter2", again) is True

# Non-ASCII passwords survive the round trip.
unicode_stored = hash_password("pässwörd-ü")
assert verify_password("pässwörd-ü", unicode_stored) is True
assert verify_password("password-u", unicode_stored) is False

# An older, cheaper hash must still verify: the cost comes from the record.
cheap = "pbkdf2_sha256$120000$" + salt_b64 + "$"
import hashlib
derived = hashlib.pbkdf2_hmac(
    "sha256", "legacy".encode("utf-8"), base64.b64decode(salt_b64), 120000
)
cheap = cheap + base64.b64encode(derived).decode("ascii")
assert verify_password("legacy", cheap) is True, (
    "verification must use the iterations recorded in the stored value"
)
assert verify_password("wrong", cheap) is False

# Malformed input is a False, not a traceback reaching the login handler.
for bad in ("", "nonsense", "pbkdf2_sha256$abc$def", "scrypt$1$aa$bb",
            "pbkdf2_sha256$100000$!!!$!!!", "a$b$c$d$e", None, 7):
    assert verify_password("hunter2", bad) is False, bad
'''),

    task(
        f"{FAMILY}-0004", FAMILY,
        prompt=(
            "Implement a Python function authorize(principal, action, "
            "resource) returning True or False for an object-level access "
            "decision. `principal` is a dict with keys id and roles (a list "
            "of strings) or None for an unauthenticated caller. `action` is "
            "'read', 'write' or 'delete'. `resource` is a dict with keys "
            "owner_id and visibility ('public' or 'private'). Deny by "
            "default: any unknown action, any principal that is None, and "
            "any role not named below yields False. An 'admin' role may do "
            "anything. An 'auditor' role may read anything but never write "
            "or delete. Any authenticated principal may read a public "
            "resource. Otherwise a principal may read, write or delete only "
            "a resource whose owner_id equals its own id. Ownership is "
            "compared by exact equality of both type and value, so a caller "
            "must not reach another principal's private resource, and 0 does "
            "not own a resource owned by False. Absent identity is never "
            "ownership: if either the principal's id or the resource's "
            "owner_id is None, they do not match even when both are None."
        ),
        validator=LOAD_CANDIDATE + require("authorize") + r'''
ADA = {"id": "u1", "roles": ["member"]}
BOB = {"id": "u2", "roles": ["member"]}
ADMIN = {"id": "u3", "roles": ["admin"]}
AUDITOR = {"id": "u4", "roles": ["auditor"]}

ADA_PRIVATE = {"owner_id": "u1", "visibility": "private"}
BOB_PRIVATE = {"owner_id": "u2", "visibility": "private"}
PUBLIC = {"owner_id": "u2", "visibility": "public"}

# An owner reaches their own resource.
for action in ("read", "write", "delete"):
    assert authorize(ADA, action, ADA_PRIVATE) is True, action

# The whole point: authentication is not authorization. Ada is a valid
# principal and still must not touch Bob's private resource.
for action in ("read", "write", "delete"):
    assert authorize(ADA, action, BOB_PRIVATE) is False, action

# Public resources are readable by any authenticated principal, but being
# able to read one confers nothing else.
assert authorize(ADA, "read", PUBLIC) is True
assert authorize(ADA, "write", PUBLIC) is False
assert authorize(ADA, "delete", PUBLIC) is False
assert authorize(BOB, "write", PUBLIC) is True

# Admin everywhere; auditor reads only.
for action in ("read", "write", "delete"):
    assert authorize(ADMIN, action, BOB_PRIVATE) is True, action
assert authorize(AUDITOR, "read", BOB_PRIVATE) is True
assert authorize(AUDITOR, "write", BOB_PRIVATE) is False
assert authorize(AUDITOR, "delete", BOB_PRIVATE) is False

# Deny by default.
assert authorize(None, "read", PUBLIC) is False
assert authorize(None, "read", ADA_PRIVATE) is False
assert authorize(ADA, "publish", ADA_PRIVATE) is False
assert authorize(ADA, "READ", ADA_PRIVATE) is False
assert authorize({"id": "u5", "roles": []}, "read", ADA_PRIVATE) is False
assert authorize({"id": "u5", "roles": ["superuser"]}, "write", ADA_PRIVATE) is False
assert authorize({"id": "u5", "roles": ["Admin"]}, "write", ADA_PRIVATE) is False

# Ownership is exact equality, not truthiness or coercion.
assert authorize({"id": 0, "roles": ["member"]},
                 "write", {"owner_id": False, "visibility": "private"}) is False
assert authorize({"id": "u1", "roles": ["member"]},
                 "write", {"owner_id": None, "visibility": "private"}) is False
assert authorize({"id": None, "roles": ["member"]},
                 "write", {"owner_id": None, "visibility": "private"}) is False
'''),

    task(
        f"{FAMILY}-0005", FAMILY,
        prompt=(
            "Implement two Python functions, issue_csrf(session_id, key) and "
            "check_csrf(session_id, token, key), for the signed "
            "double-submit CSRF pattern. issue_csrf returns a token string "
            "'<nonce>.<signature>' where nonce is fresh base64url random of "
            "at least 16 bytes without padding, and signature is the "
            "base64url HMAC-SHA256, without padding, over the ASCII bytes of "
            "session_id + '.' + nonce. Binding the signature to the session "
            "is the point: a token issued for one session must not validate "
            "under another. check_csrf recomputes the signature and returns "
            "True or False, comparing with hmac.compare_digest, and returns "
            "False -- never raises -- for a malformed token, a wrong key, or "
            "a non-string input. Two calls to issue_csrf for one session "
            "must return different tokens."
        ),
        validator=LOAD_CANDIDATE + require("issue_csrf") + require("check_csrf") + r'''
import base64

KEY = b"csrf-signing-key"
token = issue_csrf("session-a", KEY)
assert isinstance(token, str)
nonce, signature = token.split(".")
assert "=" not in token, "base64url segments must be unpadded"
assert len(base64.urlsafe_b64decode(nonce + "=" * (-len(nonce) % 4))) >= 16

assert check_csrf("session-a", token, KEY) is True

# The binding: the same token under a different session is worthless.
assert check_csrf("session-b", token, KEY) is False, (
    "a token must not validate under a session it was not issued for"
)

# Fresh nonce per issue.
other = issue_csrf("session-a", KEY)
assert other != token, "each issued token must carry a fresh nonce"
assert check_csrf("session-a", other, KEY) is True

# A token from another session, and the nonce reused with a foreign
# signature, both fail.
foreign = issue_csrf("session-b", KEY)
assert check_csrf("session-a", foreign, KEY) is False
assert check_csrf("session-a", nonce + "." + foreign.split(".")[1], KEY) is False

# Wrong key.
assert check_csrf("session-a", token, b"other-key") is False

# Malformed input returns False rather than raising into the request path.
for bad in ("", ".", "a.b", nonce, nonce + ".", "." + signature,
            token + ".extra", "!!!.???", None, 7, b"bytes"):
    assert check_csrf("session-a", bad, KEY) is False, repr(bad)

for bad_session in (None, 7, b"session-a"):
    assert check_csrf(bad_session, token, KEY) is False, repr(bad_session)
'''),

    task(
        f"{FAMILY}-0006", FAMILY,
        prompt=(
            "Implement a Python function build_set_cookie(name, value, *, "
            "max_age=None, path='/', secure=True, http_only=True, "
            "same_site='Lax') returning one Set-Cookie header value. Emit "
            "the attributes in this exact order, omitting those not "
            "requested: '<name>=<value>', then Max-Age, Path, Secure, "
            "HttpOnly, SameSite, joined by '; '. max_age must be an integer "
            "when given. same_site must be 'Strict', 'Lax' or 'None'. "
            "Because a SameSite=None cookie is ignored by browsers unless it "
            "is also Secure, raise ValueError when same_site is 'None' and "
            "secure is false. Reject a name or value that is empty, not a "
            "str, or contains a control character, space, semicolon, comma, "
            "double quote, backslash, or any of '=' in the name: a value "
            "carrying CR or LF would let a caller inject additional response "
            "headers. Raise ValueError on any violation."
        ),
        validator=LOAD_CANDIDATE + require("build_set_cookie") + r'''
assert build_set_cookie("sid", "abc123") == "sid=abc123; Path=/; Secure; HttpOnly; SameSite=Lax"
assert build_set_cookie("sid", "abc123", max_age=600) == (
    "sid=abc123; Max-Age=600; Path=/; Secure; HttpOnly; SameSite=Lax"
)
assert build_set_cookie("sid", "abc", path="/app") == (
    "sid=abc; Path=/app; Secure; HttpOnly; SameSite=Lax"
)
assert build_set_cookie("sid", "abc", secure=False, http_only=False) == (
    "sid=abc; Path=/; SameSite=Lax"
)
assert build_set_cookie("sid", "abc", same_site="Strict") == (
    "sid=abc; Path=/; Secure; HttpOnly; SameSite=Strict"
)
assert build_set_cookie("sid", "abc", same_site="None") == (
    "sid=abc; Path=/; Secure; HttpOnly; SameSite=None"
)
assert build_set_cookie("sid", "abc", max_age=0) == (
    "sid=abc; Max-Age=0; Path=/; Secure; HttpOnly; SameSite=Lax"
)

def rejects(why, *args, **kwargs):
    try:
        build_set_cookie(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError(why)

# Header injection is the reason the character rules exist.
rejects("CRLF in a value must be rejected", "sid", "abc\r\nSet-Cookie: admin=1")
rejects("a bare LF must be rejected", "sid", "abc\nX-Evil: 1")
rejects("a bare CR must be rejected", "sid", "abc\r")
rejects("CRLF in a name must be rejected", "sid\r\nX: 1", "abc")
rejects("a semicolon must be rejected", "sid", "abc; Path=/evil")
rejects("a NUL must be rejected", "sid", "abc\x00")
rejects("a space must be rejected", "sid", "a b")
rejects("a comma must be rejected", "sid", "a,b")
rejects("a quote must be rejected", "sid", 'a"b')
rejects("a backslash must be rejected", "sid", "a\\b")
rejects("'=' in a name must be rejected", "si=d", "abc")

# SameSite=None without Secure is silently dropped by browsers.
rejects("SameSite=None requires Secure", "sid", "abc", same_site="None", secure=False)

rejects("an empty name must be rejected", "", "abc")
rejects("an empty value must be rejected", "sid", "")
rejects("a non-str name must be rejected", 7, "abc")
rejects("a non-str value must be rejected", "sid", 7)
rejects("an unknown same_site must be rejected", "sid", "abc", same_site="lax")
rejects("a non-int max_age must be rejected", "sid", "abc", max_age="600")
rejects("a bool max_age must be rejected", "sid", "abc", max_age=True)
'''),

    task(
        f"{FAMILY}-0007", FAMILY,
        prompt=(
            "Implement a Python function safe_redirect(target, "
            "allowed_hosts) returning a redirect location that cannot leave "
            "the application. `allowed_hosts` is a set of permitted "
            "lowercase hostnames. Return target unchanged when it is safe, "
            "otherwise return '/'. A safe target is either a same-site "
            "relative path -- it starts with a single '/' and not a second "
            "'/' or a backslash -- or an absolute http/https URL whose "
            "hostname, lowercased and stripped of any port, is in "
            "allowed_hosts. Everything else falls back to '/': a scheme that "
            "is not http or https, a protocol-relative '//evil.example' that "
            "a browser resolves as an absolute URL, a backslash variant such "
            "as '/\\evil.example' that some clients normalise to one, any "
            "target carrying CR or LF, a target with embedded credentials, "
            "and any non-str input."
        ),
        validator=LOAD_CANDIDATE + require("safe_redirect") + r'''
ALLOWED = {"app.example", "www.app.example"}

# Relative paths are the ordinary safe case.
assert safe_redirect("/dashboard", ALLOWED) == "/dashboard"
assert safe_redirect("/a/b?c=d#e", ALLOWED) == "/a/b?c=d#e"
assert safe_redirect("/", ALLOWED) == "/"

# Absolute URLs to an allowed host, including a port and odd casing.
assert safe_redirect("https://app.example/x", ALLOWED) == "https://app.example/x"
assert safe_redirect("http://app.example/x", ALLOWED) == "http://app.example/x"
assert safe_redirect("https://APP.example/x", ALLOWED) == "https://APP.example/x"
assert safe_redirect("https://app.example:8443/x", ALLOWED) == "https://app.example:8443/x"

# The open-redirect cases all collapse to '/'.
assert safe_redirect("//evil.example", ALLOWED) == "/", (
    "a protocol-relative target is an absolute URL to a browser"
)
assert safe_redirect("//evil.example/path", ALLOWED) == "/"
assert safe_redirect("/\\evil.example", ALLOWED) == "/"
assert safe_redirect("\\\\evil.example", ALLOWED) == "/"
assert safe_redirect("https://evil.example/x", ALLOWED) == "/"
assert safe_redirect("https://app.example.evil.example/x", ALLOWED) == "/"
assert safe_redirect("https://evil.example/?next=app.example", ALLOWED) == "/"
assert safe_redirect("javascript:alert(1)", ALLOWED) == "/"
assert safe_redirect("data:text/html,<script>", ALLOWED) == "/"
assert safe_redirect("file:///etc/passwd", ALLOWED) == "/"

# Credentials let a target read as the allowed host while resolving elsewhere.
assert safe_redirect("https://app.example@evil.example/x", ALLOWED) == "/"

# Response-splitting attempts and junk.
assert safe_redirect("/ok\r\nSet-Cookie: a=1", ALLOWED) == "/"
assert safe_redirect("/ok\nX: 1", ALLOWED) == "/"
assert safe_redirect("", ALLOWED) == "/"
assert safe_redirect("dashboard", ALLOWED) == "/"
assert safe_redirect("../admin", ALLOWED) == "/"
for bad in (None, 7, b"/dashboard", ["/dashboard"]):
    assert safe_redirect(bad, ALLOWED) == "/", repr(bad)

# An empty allow-list still permits relative paths and nothing absolute.
assert safe_redirect("/dashboard", set()) == "/dashboard"
assert safe_redirect("https://app.example/x", set()) == "/"
'''),

    task(
        f"{FAMILY}-0008", FAMILY,
        prompt=(
            "Implement a Python function redact(record, secret_keys) that "
            "returns a copy of a nested structure with secret values "
            "replaced by the string '[REDACTED]'. `record` is built from "
            "dicts, lists, tuples and scalars. `secret_keys` is a set of "
            "lowercase key names. A dict entry is redacted when its key, "
            "lowercased, either is in secret_keys or contains one of them as "
            "a substring -- so 'Authorization', 'db_password' and "
            "'API_TOKEN' are all caught. Redact the whole value, whatever "
            "its type, without descending into it. Lists and tuples are "
            "traversed and keep their type. The input must not be mutated: a "
            "logging helper that edits the caller's object destroys the data "
            "the application was about to use. Keys keep their original "
            "case and dict ordering is preserved."
        ),
        validator=LOAD_CANDIDATE + require("redact") + r'''
import copy

SECRETS = {"password", "token", "authorization", "secret"}

record = {
    "user": "ada",
    "Authorization": "Bearer abc",
    "db_password": "hunter2",
    "API_TOKEN": "t-1",
    "nested": {
        "secret_key": "s-1",
        "keep": "visible",
        "deeper": [{"token": "t-2", "id": 3}, {"id": 4}],
    },
    "items": [1, "two", {"password": "p"}],
    "pair": ("a", {"token": "t-3"}),
    "count": 7,
    "nothing": None,
}
original = copy.deepcopy(record)
result = redact(record, SECRETS)

assert record == original, "redact must not mutate the caller's record"

assert result["user"] == "ada"
assert result["Authorization"] == "[REDACTED]"
assert result["db_password"] == "[REDACTED]"
assert result["API_TOKEN"] == "[REDACTED]"
assert result["count"] == 7
assert result["nothing"] is None
assert result["nested"]["secret_key"] == "[REDACTED]"
assert result["nested"]["keep"] == "visible"
assert result["nested"]["deeper"][0]["token"] == "[REDACTED]"
assert result["nested"]["deeper"][0]["id"] == 3
assert result["nested"]["deeper"][1] == {"id": 4}
assert result["items"] == [1, "two", {"password": "[REDACTED]"}]

# Tuples keep their type rather than degrading to lists.
assert isinstance(result["pair"], tuple)
assert result["pair"] == ("a", {"token": "[REDACTED]"})

# Original key casing and ordering survive.
assert list(result) == list(original)
assert "Authorization" in result and "authorization" not in result

# A secret whose value is a container is redacted whole, not descended into.
nested_secret = {"token": {"inner": "still-secret", "list": [1, 2]}}
assert redact(nested_secret, SECRETS) == {"token": "[REDACTED]"}

# Non-str keys are left alone rather than crashing the logger.
mixed = {1: "one", None: "none", "token": "t"}
assert redact(mixed, SECRETS) == {1: "one", None: "none", "token": "[REDACTED]"}

# Scalars and empty containers round-trip.
assert redact("plain", SECRETS) == "plain"
assert redact(5, SECRETS) == 5
assert redact([], SECRETS) == []
assert redact({}, SECRETS) == {}
assert redact({"a": 1}, set()) == {"a": 1}
'''),
]
