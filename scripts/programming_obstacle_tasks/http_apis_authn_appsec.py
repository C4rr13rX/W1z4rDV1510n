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

    task(
        f"{FAMILY}-0009", FAMILY,
        prompt=(
            "Implement a Python function parse_cache_control(header) parsing "
            "a Cache-Control header into a dict. Directive names are "
            "case-insensitive and appear lowercased in the result. A "
            "directive with no value maps to True; one with a value maps to "
            "that value as a str, with surrounding double quotes removed. "
            "The delta-seconds directives max-age, s-maxage, min-fresh, "
            "max-stale, stale-while-revalidate and stale-if-error map to an "
            "int instead, and a value that is not a bare run of ASCII digits "
            "is not a usable delta-seconds, so that one directive is dropped "
            "while the rest of the header still applies. Whitespace around "
            "names, values and separators is insignificant, and empty "
            "elements are skipped. When a directive repeats, the first "
            "occurrence wins. Crucially, the comma that separates directives "
            "may also appear inside a quoted field list, as in "
            "no-cache=\"Set-Cookie, Authorization\"; splitting on every "
            "comma tears that directive in half and loses the field list it "
            "names. A non-string header yields an empty dict."
        ),
        validator=LOAD_CANDIDATE + require("parse_cache_control") + r'''
assert parse_cache_control("") == {}
assert parse_cache_control("   ") == {}
for bad in (None, 7, b"no-store", ["no-store"]):
    assert parse_cache_control(bad) == {}, repr(bad)

# Valueless directives.
assert parse_cache_control("no-store") == {"no-store": True}
assert parse_cache_control("no-cache, no-store") == \
    {"no-cache": True, "no-store": True}
assert parse_cache_control("must-revalidate,proxy-revalidate,immutable") == \
    {"must-revalidate": True, "proxy-revalidate": True, "immutable": True}

# delta-seconds become ints.
assert parse_cache_control("max-age=60") == {"max-age": 60}
assert parse_cache_control("max-age=0") == {"max-age": 0}
assert parse_cache_control("MAX-AGE=60") == {"max-age": 60}
assert parse_cache_control("  max-age = 60  ") == {"max-age": 60}
assert parse_cache_control('max-age="60"') == {"max-age": 60}
assert parse_cache_control("max-age=0006") == {"max-age": 6}
assert parse_cache_control(
    "s-maxage=30, stale-while-revalidate=10, stale-if-error=5, min-fresh=2"
) == {"s-maxage": 30, "stale-while-revalidate": 10,
      "stale-if-error": 5, "min-fresh": 2}

# An unusable delta-seconds drops that directive only.
for bad in ("max-age=-1", "max-age=abc", "max-age=", "max-age=1.5",
            "max-age=+1", "max-age=1 2", "max-age=٦"):
    assert parse_cache_control(bad) == {}, bad
assert parse_cache_control("max-age=-1, no-store") == {"no-store": True}
assert parse_cache_control("no-store, max-age=nope, immutable") == \
    {"no-store": True, "immutable": True}

# max-stale is a delta-seconds directive that is also legal bare.
assert parse_cache_control("max-stale") == {"max-stale": True}
assert parse_cache_control("max-stale=120") == {"max-stale": 120}

# Quoted field lists survive intact, comma and all.
assert parse_cache_control('private="Set-Cookie"') == {"private": "Set-Cookie"}
assert parse_cache_control('no-cache="Set-Cookie, Authorization"') == \
    {"no-cache": "Set-Cookie, Authorization"}, (
        "a comma inside a quoted field list was treated as a separator"
    )
assert parse_cache_control(
    'no-store, no-cache="Set-Cookie, Authorization", max-age=0'
) == {"no-store": True,
      "no-cache": "Set-Cookie, Authorization",
      "max-age": 0}

# Unknown directives pass through as strings rather than being discarded.
assert parse_cache_control("immutable, x-vendor=7") == \
    {"immutable": True, "x-vendor": "7"}

# The first occurrence of a repeated directive wins.
assert parse_cache_control("max-age=60, max-age=120") == {"max-age": 60}
assert parse_cache_control("no-store, no-store") == {"no-store": True}

# Stray and empty elements.
assert parse_cache_control(",, no-store ,,") == {"no-store": True}
assert parse_cache_control("=60") == {}
'''),

    task(
        f"{FAMILY}-0010", FAMILY,
        prompt=(
            "Implement a Python function negotiate(accept, available) "
            "performing HTTP proactive content negotiation. `accept` is an "
            "Accept header string and `available` is a list of media types "
            "the server can produce, in server preference order. Return the "
            "chosen media type, or None when nothing is acceptable. Each "
            "entry may carry a q parameter defaulting to 1.0; q=0 means the "
            "client refuses that type. Match `*/*` and `type/*` wildcards. A "
            "candidate's quality comes from the most specific range that "
            "matches it, so `text/*;q=0.8` beats `*/*;q=0.2` for text. The "
            "highest quality wins, ties break by the order of `available`. "
            "Media types, subtypes and the q parameter name are all "
            "case-insensitive, surrounding whitespace is insignificant, and "
            "parameters other than q do not affect matching. Ignore any "
            "entry whose q is not a number in [0, 1]. A missing or empty "
            "Accept header means the client will take anything."
        ),
        validator=LOAD_CANDIDATE + require("negotiate") + r'''
AVAIL = ["application/json", "text/html", "text/plain"]

assert negotiate("application/json", AVAIL) == "application/json"
assert negotiate("text/plain", AVAIL) == "text/plain"

# No preference expressed: the server's own first choice.
assert negotiate("", AVAIL) == "application/json"
assert negotiate(None, AVAIL) == "application/json"
assert negotiate("   ", AVAIL) == "application/json"
assert negotiate("*/*", AVAIL) == "application/json"

# A subtype wildcard restricts to that type.
assert negotiate("text/*", AVAIL) == "text/html"

# Quality ordering decides between two acceptable types.
assert negotiate("text/html;q=0.3, application/json;q=0.9", AVAIL) == \
    "application/json"
assert negotiate("text/html;q=0.9, application/json;q=0.3", AVAIL) == \
    "text/html"

# q=0 is a refusal, not a low preference.
assert negotiate("*/*, application/json;q=0", AVAIL) == "text/html"
assert negotiate("application/json;q=0", ["application/json"]) is None
assert negotiate("*/*;q=0", AVAIL) is None

# Specificity selects which range supplies the quality.
assert negotiate("*/*;q=0.2, text/*;q=0.8", AVAIL) == "text/html"
assert negotiate("*/*;q=0.9, text/plain;q=0.4", AVAIL) == "application/json"
assert negotiate("text/*;q=0.9, text/plain;q=0.4", AVAIL) == "text/html"

# Case, whitespace and unrelated parameters are all insignificant.
assert negotiate("  TEXT/HTML ;  Q=0.9 , */* ; q=0.1 ", AVAIL) == "text/html"
assert negotiate("text/html;level=1;q=0.9, */*;q=0.1", AVAIL) == "text/html"

# A malformed q makes the entry unusable rather than the request a failure.
assert negotiate("application/json;q=bogus, text/html", AVAIL) == "text/html"
assert negotiate("application/json;q=7, text/html", AVAIL) == "text/html"

# Nothing on offer matches.
assert negotiate("image/png", AVAIL) is None
assert negotiate("*/*", []) is None
assert negotiate("", []) is None

# Empty list items and stray commas do not derail parsing.
assert negotiate("text/html, ,", AVAIL) == "text/html"
'''),

    task(
        f"{FAMILY}-0011", FAMILY,
        prompt=(
            "Implement a Python function conditional_status(method, headers, "
            "etag, last_modified) deciding the HTTP status for a conditional "
            "request against an existing resource. `headers` maps header "
            "names, in any case, to their raw string values. `etag` is the "
            "resource's current entity tag exactly as it would be sent, "
            "which may be weak (prefixed `W/`). Return 304 when the client's "
            "cached copy is current, 412 when a precondition fails, and 200 "
            "otherwise. If-None-Match uses the weak comparison function, so "
            "`W/\"a\"` and `\"a\"` match; it yields 304 for GET and HEAD but "
            "412 for any other method. If-Match uses the strong comparison "
            "function, so a weak tag on either side never matches, and "
            "yields 412 on mismatch. The value `*` matches any existing "
            "resource. Either header may carry a comma-separated list. "
            "If-None-Match takes precedence when both are present. "
            "If-Modified-Since is only consulted when If-None-Match is "
            "absent, and only for GET and HEAD; it is an RFC 1123 date and a "
            "malformed one is ignored."
        ),
        validator=LOAD_CANDIDATE + require("conditional_status") + r'''
ETAG = '"v1"'
MOD = "Wed, 21 Oct 2026 07:28:00 GMT"

assert conditional_status("GET", {}, ETAG, MOD) == 200

# Weak comparison for If-None-Match: a weak tag still means "unchanged".
assert conditional_status("GET", {"If-None-Match": '"v1"'}, ETAG, MOD) == 304
assert conditional_status("GET", {"If-None-Match": 'W/"v1"'}, ETAG, MOD) == 304
assert conditional_status("GET", {"If-None-Match": '"v2"'}, ETAG, MOD) == 200
assert conditional_status("HEAD", {"If-None-Match": '"v1"'}, ETAG, MOD) == 304

# A weak resource tag also compares weakly.
assert conditional_status("GET", {"If-None-Match": '"v1"'}, 'W/"v1"', MOD) == 304

# On an unsafe method a matching If-None-Match is a failed precondition,
# not a cache hit -- this is the lost-update guard.
assert conditional_status("PUT", {"If-None-Match": '"v1"'}, ETAG, MOD) == 412
assert conditional_status("DELETE", {"If-None-Match": '*'}, ETAG, MOD) == 412
assert conditional_status("PUT", {"If-None-Match": '"v2"'}, ETAG, MOD) == 200

# Strong comparison for If-Match: a weak tag on either side cannot match.
assert conditional_status("PUT", {"If-Match": '"v1"'}, ETAG, MOD) == 200
assert conditional_status("PUT", {"If-Match": 'W/"v1"'}, ETAG, MOD) == 412
assert conditional_status("PUT", {"If-Match": '"v1"'}, 'W/"v1"', MOD) == 412
assert conditional_status("PUT", {"If-Match": '"v2"'}, ETAG, MOD) == 412
assert conditional_status("PUT", {"If-Match": '*'}, ETAG, MOD) == 200

# Lists, whitespace and header-name casing.
assert conditional_status(
    "GET", {"if-none-match": '"a", W/"v1" , "b"'}, ETAG, MOD) == 304
assert conditional_status(
    "PUT", {"IF-MATCH": '"a","b"'}, ETAG, MOD) == 412
assert conditional_status(
    "PUT", {"IF-MATCH": '"a", "v1"'}, ETAG, MOD) == 200

# If-None-Match wins when both are present.
both = {"If-None-Match": '"v1"', "If-Match": '"other"'}
assert conditional_status("GET", both, ETAG, MOD) == 304

# If-Modified-Since only without If-None-Match, and only on safe methods.
assert conditional_status("GET", {"If-Modified-Since": MOD}, ETAG, MOD) == 304
later = "Thu, 22 Oct 2026 07:28:00 GMT"
earlier = "Tue, 20 Oct 2026 07:28:00 GMT"
assert conditional_status(
    "GET", {"If-Modified-Since": later}, ETAG, MOD) == 304
assert conditional_status(
    "GET", {"If-Modified-Since": earlier}, ETAG, MOD) == 200
assert conditional_status("PUT", {"If-Modified-Since": MOD}, ETAG, MOD) == 200
assert conditional_status(
    "GET", {"If-None-Match": '"v2"', "If-Modified-Since": MOD}, ETAG, MOD) == 200

# A malformed date is ignored rather than treated as a precondition failure.
assert conditional_status(
    "GET", {"If-Modified-Since": "not-a-date"}, ETAG, MOD) == 200
'''),

    task(
        f"{FAMILY}-0012", FAMILY,
        prompt=(
            "Implement a Python function preflight(origin, method, "
            "request_headers, config) deciding a CORS preflight. "
            "`request_headers` is the list of header names from "
            "Access-Control-Request-Headers. `config` is a dict with keys "
            "`allowed_origins` (a set), `allowed_methods` (a set), "
            "`allowed_headers` (a set of lowercase names), "
            "`allow_credentials` (bool) and `max_age` (int seconds). Return "
            "a dict of response headers when the preflight is approved, or "
            "None when it must be denied. Approval requires the origin to be "
            "in allowed_origins, the method to be in allowed_methods, and "
            "every requested header to be allowed; header names compare "
            "case-insensitively. On approval return "
            "Access-Control-Allow-Origin echoing the exact requested origin, "
            "Access-Control-Allow-Methods and Access-Control-Allow-Headers "
            "as comma-separated sorted lists of what was allowed for this "
            "request, Access-Control-Max-Age as a string, and Vary: Origin. "
            "Include Access-Control-Allow-Credentials only when credentials "
            "are enabled. A configuration containing the wildcard `*` in "
            "allowed_origins may only be honoured when credentials are "
            "disabled; with credentials enabled a wildcard origin must be "
            "denied rather than reflected, because the pair grants every "
            "site authenticated access."
        ),
        validator=LOAD_CANDIDATE + require("preflight") + r'''
BASE = {
    "allowed_origins": {"https://app.example"},
    "allowed_methods": {"GET", "POST", "PUT"},
    "allowed_headers": {"content-type", "authorization"},
    "allow_credentials": False,
    "max_age": 600,
}


def config(**overrides):
    merged = dict(BASE)
    merged.update(overrides)
    return merged


ok = preflight("https://app.example", "POST", ["Content-Type"], config())
assert ok is not None, "a valid preflight was denied"
assert ok["Access-Control-Allow-Origin"] == "https://app.example"
assert ok["Access-Control-Allow-Methods"] == "POST"
assert ok["Access-Control-Allow-Headers"] == "content-type"
assert ok["Access-Control-Max-Age"] == "600"
assert ok["Vary"] == "Origin"
assert "Access-Control-Allow-Credentials" not in ok

# Header names compare case-insensitively and come back sorted.
ok = preflight(
    "https://app.example", "PUT",
    ["AUTHORIZATION", "Content-Type"], config())
assert ok["Access-Control-Allow-Headers"] == "authorization, content-type"

# No requested headers means an empty allow list, not a denial.
ok = preflight("https://app.example", "GET", [], config())
assert ok is not None and ok["Access-Control-Allow-Headers"] == ""

# Denials.
assert preflight("https://evil.example", "GET", [], config()) is None
assert preflight("https://app.example", "DELETE", [], config()) is None
assert preflight(
    "https://app.example", "GET", ["X-Secret"], config()) is None
assert preflight(
    "https://app.example", "GET",
    ["Content-Type", "X-Secret"], config()) is None

# Origins compare exactly: scheme, host and port are all part of the origin.
assert preflight("http://app.example", "GET", [], config()) is None
assert preflight("https://app.example:8443", "GET", [], config()) is None
assert preflight("https://app.example/", "GET", [], config()) is None
assert preflight("", "GET", [], config()) is None
assert preflight(None, "GET", [], config()) is None

# Credentials add the header and still echo the exact origin.
creds = preflight(
    "https://app.example", "POST", ["Authorization"],
    config(allow_credentials=True))
assert creds["Access-Control-Allow-Credentials"] == "true"
assert creds["Access-Control-Allow-Origin"] == "https://app.example"

# A wildcard origin is honoured only without credentials.
wild = preflight(
    "https://anything.example", "GET", [],
    config(allowed_origins={"*"}))
assert wild is not None, "wildcard origin without credentials was denied"
assert wild["Access-Control-Allow-Origin"] in ("*", "https://anything.example")

deadly = preflight(
    "https://anything.example", "GET", [],
    config(allowed_origins={"*"}, allow_credentials=True))
assert deadly is None, (
    "wildcard origin combined with credentials must be denied, not reflected"
)

# The method is matched case-sensitively, as HTTP methods are.
assert preflight("https://app.example", "post", [], config()) is None
'''),

    task(
        f"{FAMILY}-0013", FAMILY,
        prompt=(
            "Implement a Python function is_safe_url(url) returning True "
            "only when the URL is safe for a server to fetch on a user's "
            "behalf, guarding against server-side request forgery. Require "
            "the scheme to be exactly http or https. Reject any URL "
            "carrying userinfo before the host. Reject a host that resolves "
            "on its face to a non-public address: any IPv4 or IPv6 literal "
            "that is private, loopback, link-local, unique-local, "
            "multicast, unspecified or otherwise reserved, including the "
            "cloud metadata address 169.254.169.254 and IPv4-mapped IPv6 "
            "forms of those ranges. Reject a bare integer host such as "
            "http://2130706433/, which is 127.0.0.1 written in decimal. "
            "Reject the names localhost, any name under .localhost, .local "
            "or .internal, and any empty host. Everything else, including "
            "an ordinary public hostname or public IP literal, is safe. The "
            "function must not perform DNS resolution or any network I/O."
        ),
        validator=LOAD_CANDIDATE + require("is_safe_url") + r'''
assert is_safe_url("https://example.com/path?q=1") is True
assert is_safe_url("http://example.com") is True
assert is_safe_url("https://sub.example.co.uk:8443/x") is True
assert is_safe_url("https://93.184.216.34/") is True
assert is_safe_url("https://[2606:2800:220:1:248:1893:25c8:1946]/") is True

# Scheme restriction.
for bad in ("file:///etc/passwd", "gopher://example.com/",
            "ftp://example.com/", "data:text/plain,hi",
            "//example.com/x", "javascript:alert(1)", ""):
    assert is_safe_url(bad) is False, bad

# Userinfo lets an attacker disguise the real host.
assert is_safe_url("https://example.com@169.254.169.254/") is False
assert is_safe_url("https://user:pass@example.com/") is False

# Private, loopback and link-local literals.
for bad in ("http://127.0.0.1/", "http://127.1.2.3/", "http://10.0.0.5/",
            "http://192.168.1.1/", "http://172.16.0.1/", "http://0.0.0.0/",
            "http://169.254.169.254/latest/meta-data/",
            "http://169.254.170.2/", "http://100.64.0.1/",
            "http://224.0.0.1/", "http://255.255.255.255/"):
    assert is_safe_url(bad) is False, bad

# The same ranges in IPv6, including the IPv4-mapped spelling.
for bad in ("http://[::1]/", "http://[::]/", "http://[fe80::1]/",
            "http://[fc00::1]/", "http://[fd00::1]/", "http://[ff02::1]/",
            "http://[::ffff:127.0.0.1]/", "http://[::ffff:169.254.169.254]/"):
    assert is_safe_url(bad) is False, bad

# Alternate integer spellings of the same loopback address.
assert is_safe_url("http://2130706433/") is False
assert is_safe_url("http://3232235777/") is False

# Reserved names.
for bad in ("http://localhost/", "http://LOCALHOST:8080/",
            "http://api.localhost/", "http://db.internal/",
            "http://printer.local/", "http:///path"):
    assert is_safe_url(bad) is False, bad

# A public host that merely contains a reserved name as a substring is fine.
assert is_safe_url("https://localhost.example.com/") is True
assert is_safe_url("https://internal-docs.example.com/") is True

# Non-string input is not a crash.
for bad in (None, 7, b"https://example.com/"):
    assert is_safe_url(bad) is False, repr(bad)
'''),

    task(
        f"{FAMILY}-0014", FAMILY,
        prompt=(
            "Implement a Python function verify_webhook(body, header, "
            "secret, now, tolerance) validating an inbound webhook. `body` "
            "is the exact request bytes. `header` is a signature header of "
            "the form `t=<unix seconds>,v1=<hex>` which may repeat v1 "
            "several times during a secret rotation. `secret` is bytes. The "
            "signature is the hex HMAC-SHA256, over the ASCII bytes of the "
            "timestamp, a literal '.', and the body, keyed by the secret. "
            "Return True only when the header parses, the timestamp is "
            "within `tolerance` seconds of `now` in either direction, and at "
            "least one v1 value matches. Compare digests in constant time. "
            "Return False rather than raising for any malformed header."
        ),
        validator=LOAD_CANDIDATE + require("verify_webhook") + r'''
import hashlib
import hmac

SECRET = b"whsec_test"
BODY = b'{"id":"evt_1","amount":500}'
NOW = 1788645856


def sign(body, timestamp, secret=SECRET):
    payload = f"{timestamp}.".encode("ascii") + body
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


good = f"t={NOW},v1={sign(BODY, NOW)}"
assert verify_webhook(BODY, good, SECRET, NOW, 300) is True

# Inside the window in both directions, outside it in both.
assert verify_webhook(
    BODY, f"t={NOW - 299},v1={sign(BODY, NOW - 299)}", SECRET, NOW, 300) is True
assert verify_webhook(
    BODY, f"t={NOW + 299},v1={sign(BODY, NOW + 299)}", SECRET, NOW, 300) is True
assert verify_webhook(
    BODY, f"t={NOW - 301},v1={sign(BODY, NOW - 301)}", SECRET, NOW, 300) is \
    False, "replayed an expired signature"
assert verify_webhook(
    BODY, f"t={NOW + 301},v1={sign(BODY, NOW + 301)}", SECRET, NOW, 300) is \
    False, "accepted a signature from the future"

# The timestamp is signed, so it cannot be moved forward independently.
moved = f"t={NOW},v1={sign(BODY, NOW - 5000)}"
assert verify_webhook(BODY, moved, SECRET, NOW, 300) is False, (
    "the timestamp is not covered by the signature"
)

# The body is signed, so it cannot be swapped.
assert verify_webhook(
    b'{"id":"evt_1","amount":50000}', good, SECRET, NOW, 300) is False
assert verify_webhook(BODY + b" ", good, SECRET, NOW, 300) is False

# A different secret does not verify.
assert verify_webhook(BODY, good, b"whsec_other", NOW, 300) is False

# Rotation: several v1 values, any one of which may match.
rotating = (
    f"t={NOW},v1={sign(BODY, NOW, b'old-secret')},v1={sign(BODY, NOW)}"
)
assert verify_webhook(BODY, rotating, SECRET, NOW, 300) is True
neither = (
    f"t={NOW},v1={sign(BODY, NOW, b'old')},v1={sign(BODY, NOW, b'older')}"
)
assert verify_webhook(BODY, neither, SECRET, NOW, 300) is False

# Malformed headers are rejected, not raised on.
for bad in ("", "t=,v1=", f"t={NOW}", f"v1={sign(BODY, NOW)}",
            f"t=abc,v1={sign(BODY, NOW)}", f"t={NOW},v1=zzzz",
            f"t={NOW},v1={sign(BODY, NOW)[:-1]}", "garbage",
            f"t={NOW};v1={sign(BODY, NOW)}", None, 7):
    assert verify_webhook(BODY, bad, SECRET, NOW, 300) is False, repr(bad)

# An empty body is signable like any other.
empty = f"t={NOW},v1={sign(b'', NOW)}"
assert verify_webhook(b"", empty, SECRET, NOW, 300) is True

# A zero tolerance admits only the exact second.
assert verify_webhook(BODY, good, SECRET, NOW, 0) is True
assert verify_webhook(BODY, good, SECRET, NOW + 1, 0) is False
'''),

    task(
        f"{FAMILY}-0015", FAMILY,
        prompt=(
            "Implement a Python class SessionStore guarding against session "
            "fixation. Expose create() returning a new opaque session id for "
            "an anonymous session; get(sid) returning that session's data "
            "dict or None when the id is unknown; set(sid, key, value) "
            "storing application data; authenticate(sid, user) which logs a "
            "principal in; and logout(sid). Session ids must be "
            "unguessable, generated with the secrets module, and never "
            "reused. authenticate must issue a *new* id and invalidate the "
            "old one, because an attacker who planted a known id before "
            "login would otherwise hold a valid authenticated session; it "
            "returns the new id. Application data set before login carries "
            "over to the new session. authenticate on an unknown id returns "
            "None and creates nothing. get on an authenticated session "
            "includes the key 'user'. logout invalidates the id and returns "
            "True, or False when the id was already unknown."
        ),
        validator=LOAD_CANDIDATE + require("SessionStore") + r'''
store = SessionStore()

anon = store.create()
assert isinstance(anon, str) and len(anon) >= 16, "session id is too short"
assert store.get(anon) == {} or store.get(anon) is not None

store.set(anon, "cart", ["sku-1"])
assert store.get(anon)["cart"] == ["sku-1"]
assert "user" not in store.get(anon)

# Logging in must rotate the identifier.
authed = store.authenticate(anon, "ada")
assert authed is not None, "authenticate rejected a valid session"
assert authed != anon, (
    "authenticate reused the pre-login session id, so a planted id survives "
    "the privilege change"
)
assert store.get(anon) is None, "the pre-login id is still valid after login"
assert store.get(authed)["user"] == "ada"
assert store.get(authed)["cart"] == ["sku-1"], "data did not carry over"

# Ids are unguessable and never repeat.
seen = {anon, authed}
for _ in range(200):
    sid = store.create()
    assert sid not in seen, "session id was reused"
    seen.add(sid)
assert len(set(len(s) for s in seen)) == 1, "id length varies"

# Unknown ids do nothing.
assert store.get("does-not-exist") is None
assert store.authenticate("does-not-exist", "eve") is None
assert store.get("does-not-exist") is None
assert store.logout("does-not-exist") is False

# Re-authenticating rotates again.
again = store.authenticate(authed, "ada")
assert again is not None and again not in (authed, anon)
assert store.get(authed) is None

# Logout invalidates exactly once.
assert store.logout(again) is True
assert store.get(again) is None
assert store.logout(again) is False

# Sessions are independent.
one, two = store.create(), store.create()
store.set(one, "k", 1)
assert store.get(two).get("k") is None

# The generator is the secrets module, not random.
with open(RESPONSE_PATH, encoding="utf-8") as handle:
    source = handle.read()
assert "secrets" in source, "session ids must come from the secrets module"
assert "random.random" not in source and "random.randint" not in source
'''),

    task(
        f"{FAMILY}-0016", FAMILY,
        prompt=(
            "Implement a Python function has_scope(granted, required) "
            "deciding whether an access token's scopes authorise an "
            "operation. `granted` is a space-delimited scope string and "
            "`required` is a single scope. Scopes are colon-delimited "
            "hierarchies such as repo:issues:read. A granted scope whose "
            "final segment is `*` covers everything beneath it, so `repo:*` "
            "covers repo:issues:read, and a bare `*` covers everything. "
            "Matching is on whole segments only: `repo:read` must not "
            "authorise `repo:read_write`, and `repo` alone does not "
            "authorise `repo:read`. Return a bool. An empty granted string, "
            "an empty required scope, or a non-string argument authorises "
            "nothing."
        ),
        validator=LOAD_CANDIDATE + require("has_scope") + r'''
assert has_scope("repo:read", "repo:read") is True
assert has_scope("repo:read repo:write", "repo:write") is True
assert has_scope("  repo:read   repo:write  ", "repo:write") is True

# Wildcards cover the subtree beneath them.
assert has_scope("repo:*", "repo:read") is True
assert has_scope("repo:*", "repo:issues:read") is True
assert has_scope("*", "anything:at:all") is True
assert has_scope("repo:issues:*", "repo:issues:read") is True

# ... but not a sibling subtree, and not the bare prefix itself.
assert has_scope("repo:*", "billing:read") is False
assert has_scope("repo:issues:*", "repo:pulls:read") is False

# Segment boundaries are the whole point: a prefix match authorises far
# more than was granted.
assert has_scope("repo:read", "repo:read_write") is False, (
    "matched on a text prefix rather than on a segment boundary"
)
assert has_scope("repo", "repo:read") is False
assert has_scope("rep:*", "repo:read") is False
assert has_scope("repo:read", "repo:read:extra") is False

# A wildcard must be a whole segment, not a suffix character.
assert has_scope("repo:re*", "repo:read") is False

# More specific granted scopes do not imply less specific ones.
assert has_scope("repo:issues:read", "repo:issues") is False
assert has_scope("repo:issues:read", "repo") is False

# Empty and malformed inputs authorise nothing.
for granted, required in (
    ("", "repo:read"), ("   ", "repo:read"), ("repo:read", ""),
    ("repo:read", "   "), (None, "repo:read"), ("repo:read", None),
    (7, "repo:read"), ("repo:read", 7), ("", ""),
):
    assert has_scope(granted, required) is False, (granted, required)

# Duplicated and irrelevant scopes are harmless.
assert has_scope("a b repo:read repo:read c", "repo:read") is True
assert has_scope("a b c", "repo:read") is False
'''),

    task(
        f"{FAMILY}-0017", FAMILY,
        prompt=(
            "Implement a Python class IdempotentStore supporting safe "
            "client retries. Also define an exception class "
            "IdempotencyConflict. Expose execute(key, fingerprint, fn): "
            "`key` is the client's Idempotency-Key, `fingerprint` is a "
            "hash of the request the key was first used with, and `fn` is a "
            "zero-argument callable performing the side effect. The first "
            "call for a key invokes fn, stores the result against the "
            "fingerprint, and returns it. A later call with the same key and "
            "the same fingerprint must return the stored result without "
            "invoking fn again -- that is the entire point, so a retried "
            "payment charges once. A call reusing the key with a different "
            "fingerprint must raise IdempotencyConflict without invoking fn, "
            "because the client is asking for a different operation under a "
            "key that already means something else. If fn raises, nothing is "
            "recorded and a subsequent retry runs it again. A key that is "
            "None or empty means the client requested no idempotency: run fn "
            "every time and store nothing."
        ),
        validator=LOAD_CANDIDATE + require("IdempotentStore")
        + require("IdempotencyConflict") + r'''
class Counter:
    def __init__(self, value="ok"):
        self.calls = 0
        self.value = value

    def __call__(self):
        self.calls += 1
        return self.value


store = IdempotentStore()

charge = Counter({"id": "ch_1"})
first = store.execute("key-1", "fp-a", charge)
assert first == {"id": "ch_1"} and charge.calls == 1

# The retry is the whole contract: same key, same request, no second charge.
again = store.execute("key-1", "fp-a", charge)
assert again == {"id": "ch_1"}, "retry did not return the recorded result"
assert charge.calls == 1, "a retried request performed the side effect twice"

# Same key, different request: a conflict, and fn must not run.
other = Counter({"id": "ch_2"})
try:
    store.execute("key-1", "fp-b", other)
except IdempotencyConflict:
    pass
else:
    raise AssertionError("reusing a key for a different request was allowed")
assert other.calls == 0, "the conflicting request still ran"

# The original mapping survives the conflict.
assert store.execute("key-1", "fp-a", charge) == {"id": "ch_1"}
assert charge.calls == 1

# Distinct keys are independent.
second = Counter({"id": "ch_3"})
assert store.execute("key-2", "fp-a", second) == {"id": "ch_3"}
assert second.calls == 1

# A failure is not recorded, so the retry genuinely retries.
class Boom(Exception):
    pass


class Flaky:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls == 1:
            raise Boom("transient")
        return "recovered"


flaky = Flaky()
try:
    store.execute("key-3", "fp-c", flaky)
except Boom:
    pass
else:
    raise AssertionError("the failure was swallowed")
assert store.execute("key-3", "fp-c", flaky) == "recovered"
assert flaky.calls == 2
# And the recovered result is now the recorded one.
assert store.execute("key-3", "fp-c", flaky) == "recovered"
assert flaky.calls == 2

# No key means no idempotency rather than one shared bucket.
for absent in (None, ""):
    passthrough = Counter("v")
    assert store.execute(absent, "fp-x", passthrough) == "v"
    assert store.execute(absent, "fp-y", passthrough) == "v"
    assert passthrough.calls == 2, "an absent key was treated as idempotent"

# A falsy but legitimate result is still a recorded result.
zero = Counter(0)
assert store.execute("key-4", "fp-d", zero) == 0
assert store.execute("key-4", "fp-d", zero) == 0
assert zero.calls == 1, "a falsy result was treated as absent and recomputed"

none_result = Counter(None)
assert store.execute("key-5", "fp-e", none_result) is None
assert store.execute("key-5", "fp-e", none_result) is None
assert none_result.calls == 1

assert issubclass(IdempotencyConflict, Exception)
'''),

    task(
        f"{FAMILY}-0018", FAMILY,
        prompt=(
            "Implement a Python function parse_basic_auth(header) parsing an "
            "HTTP Basic Authorization header. Return a (username, password) "
            "tuple of str, or None when the header is absent or malformed. "
            "The scheme token is compared case-insensitively and is followed "
            "by one or more spaces and a single base64 token; any trailing "
            "token makes the header malformed. The decoded credentials are "
            "UTF-8 and split at the FIRST colon: a username may not contain "
            "a colon but a password certainly may, so splitting anywhere "
            "else silently truncates real passwords. Reject a header with no "
            "colon, invalid base64, or bytes that are not valid UTF-8. Both "
            "the username and the password may legitimately be empty. Never "
            "raise."
        ),
        validator=LOAD_CANDIDATE + require("parse_basic_auth") + r'''
import base64


def header(raw, scheme="Basic"):
    token = base64.b64encode(raw.encode("utf-8")).decode("ascii")
    return f"{scheme} {token}"


assert parse_basic_auth(header("ada:secret")) == ("ada", "secret")

# The scheme token is case-insensitive.
for scheme in ("basic", "BASIC", "BaSiC"):
    assert parse_basic_auth(header("ada:secret", scheme)) == ("ada", "secret")

# A colon is legal in a password and must survive intact.
assert parse_basic_auth(header("ada:a:b:c")) == ("ada", "a:b:c")
assert parse_basic_auth(header("ada:http://x:8080/")) == \
    ("ada", "http://x:8080/")

# Either half may be empty.
assert parse_basic_auth(header("ada:")) == ("ada", "")
assert parse_basic_auth(header(":secret")) == ("", "secret")
assert parse_basic_auth(header(":")) == ("", "")

# Non-ASCII credentials round-trip through UTF-8.
assert parse_basic_auth(header("adé:pässwörd")) == ("adé", "pässwörd")

# Extra spaces between scheme and token are insignificant.
token = base64.b64encode(b"ada:secret").decode("ascii")
assert parse_basic_auth(f"Basic    {token}") == ("ada", "secret")
assert parse_basic_auth(f"  Basic {token}  ") == ("ada", "secret")

# Malformed headers yield None rather than an exception.
bad_utf8 = base64.b64encode(b"\xff\xfe:pw").decode("ascii")
for bad in (
    "", "Basic", "Basic ", "Bearer " + token, "Basic !!!!",
    "Basic " + base64.b64encode(b"nocolon").decode("ascii"),
    f"Basic {token} extra", "Basic " + bad_utf8,
    token, "Basicx " + token, None, 7, b"Basic " + token.encode("ascii"),
):
    assert parse_basic_auth(bad) is None, repr(bad)

# A base64 token whose padding is wrong is malformed, not silently trimmed.
assert parse_basic_auth("Basic YWRhOnNlY3JldA") in (None, ("ada", "secret"))
assert parse_basic_auth("Basic ~~~~~~~~") is None
'''),

    task(
        f"{FAMILY}-0019", FAMILY,
        prompt=(
            "Implement a Python function build_csp(policy) serialising a "
            "Content-Security-Policy header. `policy` maps directive names "
            "to a list of source expressions. Return the header value: "
            "directives sorted alphabetically, each rendered as the name "
            "followed by its sources in the given order, joined with '; '. "
            "The keywords self, none, unsafe-inline, unsafe-eval and "
            "strict-dynamic, and any nonce-... or sha256-/sha384-/sha512-... "
            "expression, must be emitted single-quoted; an unquoted `self` "
            "is a hostname, not the keyword, and silently widens the policy. "
            "Accept them written with or without quotes in the input. Raise "
            "ValueError when: default-src is absent; a directive name is not "
            "lowercase letters, digits and hyphens starting with a letter; a "
            "directive has an empty source list; `none` appears alongside "
            "any other source; or a source contains whitespace, a semicolon "
            "or a comma, since each of those would inject an unintended "
            "directive into the header."
        ),
        validator=LOAD_CANDIDATE + require("build_csp") + r'''
out = build_csp({"default-src": ["self"]})
assert out == "default-src 'self'", out

out = build_csp({
    "default-src": ["self"],
    "img-src": ["self", "https://cdn.example.com", "data:"],
    "script-src": ["self", "nonce-abc123", "strict-dynamic"],
})
assert out == (
    "default-src 'self'; "
    "img-src 'self' https://cdn.example.com data:; "
    "script-src 'self' 'nonce-abc123' 'strict-dynamic'"
), out

# Directives are sorted; source order within a directive is preserved.
out = build_csp({"style-src": ["b.example", "a.example"],
                 "default-src": ["none"],
                 "connect-src": ["self"]})
assert out == (
    "connect-src 'self'; default-src 'none'; "
    "style-src b.example a.example"
), out

# Keywords are accepted pre-quoted and are not double-quoted.
assert build_csp({"default-src": ["'self'", "'unsafe-inline'"]}) == \
    "default-src 'self' 'unsafe-inline'"
assert build_csp({"default-src": ["self"], "script-src": ["'unsafe-eval'"]}) \
    == "default-src 'self'; script-src 'unsafe-eval'"

# Hash sources are quoted, ordinary hosts and schemes are not.
out = build_csp({"default-src": ["self"],
                 "script-src": ["sha256-YWJj", "https://x.example", "blob:"]})
assert out.endswith("script-src 'sha256-YWJj' https://x.example blob:"), out

# A host that merely starts with a keyword's letters is still a host.
assert build_csp({"default-src": ["selfhosted.example"]}) == \
    "default-src selfhosted.example"


def rejects(policy, why):
    try:
        build_csp(policy)
    except ValueError:
        return
    raise AssertionError(why)


rejects({"img-src": ["self"]}, "a policy without default-src was accepted")
rejects({"default-src": []}, "an empty source list was accepted")
rejects({"default-src": ["none", "self"]},
        "'none' was accepted alongside another source")
rejects({"default-src": ["self"], "img-src": ["self", "none"]},
        "'none' was accepted alongside another source")
rejects({"default-src": ["self"], "Img-Src": ["self"]},
        "an upper-case directive name was accepted")
rejects({"default-src": ["self"], "img src": ["self"]},
        "a directive name with a space was accepted")
rejects({"default-src": ["self"], "9-src": ["self"]},
        "a directive name starting with a digit was accepted")

# Header injection through a source expression.
rejects({"default-src": ["self; script-src *"]}, "a semicolon was accepted")
rejects({"default-src": ["a.example, *"]}, "a comma was accepted")
rejects({"default-src": ["a.example b.example"]}, "a space was accepted")
rejects({"default-src": ["a.example\nb"]}, "a newline was accepted")
rejects({"default-src": ["a.example\tb"]}, "a tab was accepted")
'''),

    task(
        f"{FAMILY}-0020", FAMILY,
        prompt=(
            "Implement two Python functions, encode_cursor(state, key) and "
            "decode_cursor(cursor, key), providing tamper-evident pagination "
            "cursors. `state` is a dict of JSON-serialisable scalars such as "
            "{'after_id': 412, 'sort': 'created_at'}, and `key` is bytes. "
            "encode_cursor returns an opaque ASCII string that is safe in a "
            "URL query and carries no '=' padding. decode_cursor returns the "
            "original dict. A cursor is authenticated with HMAC-SHA256 under "
            "`key` and the digest is compared in constant time, so a client "
            "cannot edit the state to page into rows it may not see. "
            "decode_cursor raises ValueError for a cursor that has been "
            "modified, was minted under a different key, or is not a "
            "well-formed cursor at all. Cursors carry a version marker so a "
            "future format change is detectable rather than "
            "misinterpreted, and encoding is deterministic: the same state "
            "and key always produce the same cursor regardless of the "
            "insertion order of the dict."
        ),
        validator=LOAD_CANDIDATE + require("encode_cursor")
        + require("decode_cursor") + r'''
import string

KEY = b"cursor-key-1"
STATE = {"after_id": 412, "sort": "created_at", "desc": True, "q": None}

cursor = encode_cursor(STATE, KEY)
assert isinstance(cursor, str) and cursor
assert decode_cursor(cursor, KEY) == STATE

# Opaque, URL-safe, unpadded.
allowed = set(string.ascii_letters + string.digits + "-_.~")
assert set(cursor) <= allowed, f"cursor is not URL-safe: {cursor!r}"
assert "=" not in cursor and "+" not in cursor and "/" not in cursor
assert "after_id" not in cursor and "created_at" not in cursor

# Deterministic, and independent of dict insertion order.
assert encode_cursor(STATE, KEY) == cursor
reordered = {"q": None, "desc": True, "sort": "created_at", "after_id": 412}
assert encode_cursor(reordered, KEY) == cursor, "encoding is order-dependent"

# Round-trips of the ordinary shapes.
for state in ({}, {"a": 1}, {"a": ""}, {"n": -3}, {"f": 1.5},
              {"s": "unicode ✓ text"}, {"deep": "x" * 500}):
    assert decode_cursor(encode_cursor(state, KEY), KEY) == state


def rejects(value, why, key=KEY):
    try:
        decode_cursor(value, key)
    except ValueError:
        return
    raise AssertionError(why)


# A different key must not validate.
rejects(cursor, "a cursor minted under another key was accepted",
        key=b"other-key")

# Every single-character edit must be detected.
detected = 0
for index in range(len(cursor)):
    swap = "A" if cursor[index] != "A" else "B"
    tampered = cursor[:index] + swap + cursor[index + 1:]
    try:
        decode_cursor(tampered, KEY)
    except ValueError:
        detected += 1
assert detected == len(cursor), (
    f"only {detected} of {len(cursor)} single-character edits were rejected"
)

# Truncation and extension.
rejects(cursor[:-1], "a truncated cursor was accepted")
rejects(cursor[1:], "a truncated cursor was accepted")
rejects(cursor + "A", "an extended cursor was accepted")

# Not a cursor at all.
for bad in ("", "   ", "!!!!", "a", "....", None, 7, b"abc", ["x"]):
    rejects(bad, f"non-cursor input was accepted: {bad!r}")

# Two different states do not collide.
assert encode_cursor({"after_id": 1}, KEY) != encode_cursor({"after_id": 2}, KEY)

# The version marker distinguishes formats rather than being decoded blindly.
import base64
raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
assert b"v1" in raw[:8] or raw[:1] == b"\x01", (
    "cursor carries no recognisable version marker"
)
'''),

    task(
        f"{FAMILY}-0021", FAMILY,
        prompt=(
            "Implement two Python functions supporting OAuth PKCE: "
            "make_challenge(verifier, method='S256') and "
            "verify_pkce(verifier, challenge, method). A code verifier is "
            "43 to 128 characters drawn only from the unreserved set "
            "A-Z a-z 0-9 hyphen period underscore tilde; anything else, "
            "including a non-string, must raise ValueError from either "
            "function. For method 'S256' the challenge is the base64url "
            "encoding, without '=' padding, of the SHA-256 digest of the "
            "verifier's ASCII bytes. For method 'plain' the challenge is the "
            "verifier itself. Any other method raises ValueError, so a "
            "client cannot downgrade to an unsupported or absent transform. "
            "verify_pkce returns a bool and compares in constant time."
        ),
        validator=LOAD_CANDIDATE + require("make_challenge")
        + require("verify_pkce") + r'''
import base64
import hashlib

VERIFIER = "a" * 43


def expected(verifier):
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


challenge = make_challenge(VERIFIER)
assert challenge == expected(VERIFIER), challenge
assert "=" not in challenge, "challenge carries base64 padding"
assert "+" not in challenge and "/" not in challenge, (
    "challenge uses the standard base64 alphabet, not base64url"
)
assert verify_pkce(VERIFIER, challenge, "S256") is True
assert verify_pkce("b" * 43, challenge, "S256") is False

# A verifier that exercises the alphabet where base64 and base64url differ.
tricky = "~-._" + "Zz9" * 13
assert len(tricky) == 43
assert make_challenge(tricky) == expected(tricky)
assert verify_pkce(tricky, make_challenge(tricky), "S256") is True

# 'plain' is a direct comparison.
assert make_challenge(VERIFIER, "plain") == VERIFIER
assert verify_pkce(VERIFIER, VERIFIER, "plain") is True
assert verify_pkce(VERIFIER, "b" * 43, "plain") is False

# The transform is not interchangeable: an S256 challenge must not verify
# under 'plain', which is exactly the downgrade an attacker wants.
assert verify_pkce(VERIFIER, challenge, "plain") is False

# Boundary lengths.
assert make_challenge("x" * 43) == expected("x" * 43)
assert make_challenge("x" * 128) == expected("x" * 128)


def rejects(fn, args, why):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError(why)


for bad in ("x" * 42, "x" * 129, "", "x" * 43 + " ", "a" * 42 + "+",
            "a" * 42 + "/", "a" * 42 + "=", "a" * 42 + "%",
            "a" * 42 + "\n", "adé" + "a" * 40, None, 7, b"a" * 43):
    rejects(make_challenge, (bad,), f"invalid verifier accepted: {bad!r}")
    rejects(verify_pkce, (bad, challenge, "S256"),
            f"invalid verifier accepted: {bad!r}")

# An unknown or absent method is a hard error, never a silent fallback.
for method in ("S512", "plaintext", "", "s256", "PLAIN", None, 7):
    rejects(make_challenge, (VERIFIER, method),
            f"unsupported method accepted: {method!r}")
    rejects(verify_pkce, (VERIFIER, challenge, method),
            f"unsupported method accepted: {method!r}")

# A non-string challenge is a failed verification, not a crash.
for bad_challenge in (None, 7, b"abc"):
    assert verify_pkce(VERIFIER, bad_challenge, "S256") is False
'''),

    task(
        f"{FAMILY}-0022", FAMILY,
        prompt=(
            "Implement a Python function canonical_query(pairs) producing "
            "the canonical query string used as input to a request "
            "signature. `pairs` is a sequence of (key, value) str pairs, "
            "possibly with repeated keys. Percent-encode each key and each "
            "value independently: the unreserved characters A-Z a-z 0-9 "
            "hyphen period underscore tilde are literal, and every other "
            "character is encoded as one %XX group per UTF-8 byte using "
            "upper-case hexadecimal. A space is %20, never '+', because a "
            "signer and a verifier that disagree on that one character "
            "reject every request containing a space. Sort the encoded "
            "pairs by encoded key, then by encoded value, comparing as "
            "byte strings. Join as key=value with '&', always emitting the "
            "'=' even when the value is empty. An empty input gives an "
            "empty string."
        ),
        validator=LOAD_CANDIDATE + require("canonical_query") + r'''
assert canonical_query([]) == ""
assert canonical_query([("a", "1")]) == "a=1"
assert canonical_query([("b", "2"), ("a", "1")]) == "a=1&b=2"

# Empty values keep their '='.
assert canonical_query([("a", "")]) == "a="
assert canonical_query([("", "")]) == "="

# Repeated keys are preserved and ordered by value.
assert canonical_query([("a", "2"), ("a", "1")]) == "a=1&a=2"
assert canonical_query([("a", "b"), ("a", "B")]) == "a=B&a=b"

# Space is %20. '+' is itself a character to encode, not a space.
assert canonical_query([("q", "hello world")]) == "q=hello%20world"
assert canonical_query([("q", "a+b")]) == "q=a%2Bb"

# Unreserved characters stay literal; reserved ones do not.
assert canonical_query([("k", "-._~")]) == "k=-._~"
assert canonical_query([("k", "aZ09")]) == "k=aZ09"
assert canonical_query([("k", "/")]) == "k=%2F"
assert canonical_query([("k", ":")]) == "k=%3A"
assert canonical_query([("k", "=")]) == "k=%3D"
assert canonical_query([("k", "&")]) == "k=%26"
assert canonical_query([("k", "?")]) == "k=%3F"
assert canonical_query([("k", "*")]) == "k=%2A"
assert canonical_query([("k", "%")]) == "k=%25"

# Hex digits are upper-case: %2f and %2F sign differently.
assert canonical_query([("k", "\x1f")]) == "k=%1F"

# Non-ASCII is encoded per UTF-8 byte.
assert canonical_query([("k", "é")]) == "k=%C3%A9"
assert canonical_query([("k", "✓")]) == "k=%E2%9C%93"
assert canonical_query([("é", "1")]) == "%C3%A9=1"

# Keys are encoded before sorting, so ordering is over the encoded bytes.
assert canonical_query([("a b", "1"), ("a-b", "2")]) == "a%20b=1&a-b=2"

# Sorting is by key first, then value.
assert canonical_query(
    [("b", "1"), ("a", "z"), ("a", "a")]) == "a=a&a=z&b=1"

# Byte ordering, not locale or case-insensitive ordering.
assert canonical_query([("Z", "1"), ("a", "2")]) == "Z=1&a=2"

# A realistic signing payload.
assert canonical_query([
    ("Action", "ListUsers"),
    ("Version", "2026-09-05"),
    ("Filter", "name eq 'ada lovelace'"),
]) == (
    "Action=ListUsers&"
    "Filter=name%20eq%20%27ada%20lovelace%27&"
    "Version=2026-09-05"
)
'''),

    task(
        f"{FAMILY}-0023", FAMILY,
        prompt=(
            "Implement a Python function effective_method(method, headers, "
            "form) resolving an HTTP method override. `headers` maps header "
            "names in any case to values, and `form` is a dict of submitted "
            "form fields. An override may be requested by the "
            "X-HTTP-Method-Override header or by a `_method` form field, "
            "with the header taking precedence. Honour an override only "
            "when the actual request method is POST, and only when the "
            "requested target is PUT, PATCH or DELETE. Honouring an "
            "override on GET would let a plain hyperlink or an image tag "
            "perform a deletion, so a GET is always returned unchanged. "
            "Return the uppercased effective method; when no override "
            "applies return the original method uppercased. The requested "
            "override is matched case-insensitively after stripping "
            "surrounding whitespace, and an unrecognised or empty override "
            "is ignored rather than an error."
        ),
        validator=LOAD_CANDIDATE + require("effective_method") + r'''
assert effective_method("GET", {}, {}) == "GET"
assert effective_method("POST", {}, {}) == "POST"
assert effective_method("post", {}, {}) == "POST"

# The supported overrides, from either source.
for target in ("PUT", "PATCH", "DELETE"):
    assert effective_method(
        "POST", {"X-HTTP-Method-Override": target}, {}) == target
    assert effective_method("POST", {}, {"_method": target}) == target
    assert effective_method(
        "POST", {"x-http-method-override": target.lower()}, {}) == target
    assert effective_method("POST", {}, {"_method": f"  {target}  "}) == target

# The header wins over the form field.
assert effective_method(
    "POST", {"X-HTTP-Method-Override": "PUT"}, {"_method": "DELETE"}) == "PUT"

# Only POST may be overridden. This is the defect that turns a link into a
# deletion, so every other method is returned untouched.
for actual in ("GET", "HEAD", "PUT", "DELETE", "PATCH", "OPTIONS"):
    assert effective_method(
        actual, {"X-HTTP-Method-Override": "DELETE"}, {}) == actual, actual
    assert effective_method(actual, {}, {"_method": "DELETE"}) == actual, actual

# Only the three unsafe targets are reachable by override.
for target in ("GET", "HEAD", "OPTIONS", "TRACE", "CONNECT", "POST",
               "", "   ", "BOGUS", "PUT;DROP", "DELETE ALL"):
    assert effective_method(
        "POST", {"X-HTTP-Method-Override": target}, {}) == "POST", target
    assert effective_method("POST", {}, {"_method": target}) == "POST", target

# A present but useless header falls through to nothing, not to the form.
assert effective_method(
    "POST", {"X-HTTP-Method-Override": "BOGUS"}, {"_method": "DELETE"}) == \
    "POST", "an unrecognised header override fell through to the form field"

# Non-string overrides are ignored rather than raising.
for bad in (None, 7, ["DELETE"], b"DELETE"):
    assert effective_method(
        "POST", {"X-HTTP-Method-Override": bad}, {}) == "POST", repr(bad)
    assert effective_method("POST", {}, {"_method": bad}) == "POST", repr(bad)

# Unrelated headers and fields are inert.
assert effective_method(
    "POST", {"X-Method": "DELETE"}, {"method": "DELETE"}) == "POST"
'''),

    task(
        f"{FAMILY}-0024", FAMILY,
        prompt=(
            "Implement a Python function request_framing(headers) deciding "
            "how an HTTP/1.1 request body is delimited. `headers` is a list "
            "of (name, value) pairs in wire order, preserving duplicates and "
            "original casing. Return ('chunked', None) when the body is "
            "chunked, or ('length', n) when it is delimited by a byte "
            "count; a request with neither header has an empty body, "
            "('length', 0). Raise ValueError for any framing a front end and "
            "a back end could resolve differently, because that "
            "disagreement is request smuggling: both Content-Length and "
            "Transfer-Encoding present; Content-Length repeated at all; a "
            "Transfer-Encoding whose final coding is not chunked; and a "
            "Content-Length that, after stripping the optional leading and "
            "trailing spaces and tabs every header value may carry, is not a "
            "bare run of ASCII digits -- which excludes a leading plus or "
            "minus, embedded or other whitespace, and any non-digit. "
            "Header names are "
            "case-insensitive, and the value of a Transfer-Encoding may be a "
            "comma-separated list whose codings are matched "
            "case-insensitively after stripping whitespace."
        ),
        validator=LOAD_CANDIDATE + require("request_framing") + r'''
assert request_framing([]) == ("length", 0)
assert request_framing([("Host", "example.com")]) == ("length", 0)
assert request_framing([("Content-Length", "0")]) == ("length", 0)
assert request_framing([("Content-Length", "42")]) == ("length", 42)
assert request_framing([("content-length", "42")]) == ("length", 42)
assert request_framing([("CONTENT-LENGTH", "42")]) == ("length", 42)

# A surrounding-whitespace value is ordinary header folding, and the digits
# themselves are what must be bare.
assert request_framing([("Content-Length", " 42 ")]) == ("length", 42)

assert request_framing([("Transfer-Encoding", "chunked")]) == \
    ("chunked", None)
assert request_framing([("transfer-encoding", "CHUNKED")]) == \
    ("chunked", None)
assert request_framing([("Transfer-Encoding", "gzip, chunked")]) == \
    ("chunked", None)
assert request_framing([("Transfer-Encoding", " gzip ,  chunked ")]) == \
    ("chunked", None)

# Duplicate Transfer-Encoding headers combine in wire order.
assert request_framing([
    ("Transfer-Encoding", "gzip"), ("Transfer-Encoding", "chunked"),
]) == ("chunked", None)


def rejects(headers, why):
    try:
        request_framing(headers)
    except ValueError:
        return
    raise AssertionError(why)


# The classic CL.TE / TE.CL desynchronisation.
rejects([("Content-Length", "6"), ("Transfer-Encoding", "chunked")],
        "Content-Length and Transfer-Encoding were accepted together")
rejects([("Transfer-Encoding", "chunked"), ("Content-Length", "6")],
        "Content-Length and Transfer-Encoding were accepted together")
rejects([("Content-Length", "0"), ("transfer-encoding", "chunked")],
        "a zero Content-Length beside Transfer-Encoding was accepted")

# Two byte counts, agreeing or not, are still two answers.
rejects([("Content-Length", "6"), ("Content-Length", "5")],
        "conflicting Content-Length headers were accepted")
rejects([("Content-Length", "6"), ("Content-Length", "6")],
        "a repeated Content-Length was accepted")
rejects([("Content-Length", "6"), ("content-length", "6")],
        "a repeated Content-Length differing only in case was accepted")

# Chunked must be the final coding, or the body has no known end.
rejects([("Transfer-Encoding", "chunked, gzip")],
        "chunked was accepted as a non-final coding")
rejects([("Transfer-Encoding", "gzip")],
        "a non-chunked Transfer-Encoding was accepted")
rejects([("Transfer-Encoding", "")], "an empty Transfer-Encoding was accepted")
rejects([("Transfer-Encoding", "chunk")],
        "a coding that merely resembles chunked was accepted")
rejects([("Transfer-Encoding", "chunked, chunked")],
        "chunked applied twice was accepted")

# A Content-Length that any two parsers might read differently.
for bad in ("+6", "-6", "6 6", "0x6", "6.0", "six", "",
            " ", "6,6", "0006abc", "\r6", "6\n", "٦"):
    rejects([("Content-Length", bad)],
            f"a malformed Content-Length was accepted: {bad!r}")

# A tab is ordinary optional whitespace, like a space.
assert request_framing([("Content-Length", "\t42 ")]) == ("length", 42)

# Leading zeros alone are unambiguous.
assert request_framing([("Content-Length", "0006")]) == ("length", 6)
'''),
]
