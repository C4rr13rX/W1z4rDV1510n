"""Shared validator fragments for authored obstacle tasks.

Underscore-prefixed so `load_authored_tasks` skips it: this module holds
source snippets, not tasks.

Validators run under `python -I -S`, so only the standard library is
importable. That is deliberate rather than incidental -- a task whose verdict
depends on a package installed on one machine and not another is exactly the
flaky case the acceptance contract refuses to admit.
"""

from __future__ import annotations

#: Load the candidate as a module. A candidate that does not parse raises
#: SyntaxError with the candidate's own filename, which the run harness
#: attributes to the candidate rather than to itself.
LOAD_CANDIDATE = """
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("candidate", RESPONSE_PATH)
candidate = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(candidate)
"""


#: Guards for the moment a validator first touches something the candidate
#: produced, rather than something it merely defined.
#:
#: `require` proves a name exists. It cannot prove the name is the right KIND
#: of thing, and the gap between those matters more than it looks: the run
#: harness blames the candidate only for exceptions whose traceback passes
#: through the candidate's file. A candidate whose `LRUCache` is a function
#: with a `pass` body returns None, and the validator's next line raises
#: `AttributeError: 'NoneType' object has no attribute 'put'` in validator
#: frames alone -- scored `validator_error`, which the contract treats as a
#: harness fault that blocks admission, when it is an ordinary wrong answer.
#:
#: Measured 2026-09-09 by `scripts/audit_obstacle_validator_attribution.py`:
#: 88 of 225 authored tasks did this against a stub candidate.
SHAPE_GUARDS = """
def built(value, what):
    \"\"\"The candidate produced an object here, not None.\"\"\"
    assert value is not None, f'{what} returned None'
    return value


def having(value, *names, what='the object'):
    \"\"\"...and it carries the members the prompt said it would.\"\"\"
    built(value, what)
    for name in names:
        assert hasattr(value, name), f'{what} has no {name}'
    return value


def iterating(value, what):
    \"\"\"...and it can actually be iterated.\"\"\"
    built(value, what)
    try:
        iter(value)
    except TypeError:
        raise AssertionError(f'{what} is not iterable') from None
    return value


def returning(function, what):
    \"\"\"Wrap a candidate callable so EVERY call proves it returned something.

    Guarding only the first call is not enough when a validator calls the
    same function repeatedly and reads each result: the second call is just
    as capable of raising in validator frames as the first.
    \"\"\"
    def checked(*args, **kwargs):
        return built(function(*args, **kwargs), what)
    return checked
"""


def require(name: str) -> str:
    """Assert the candidate exposes the exact public name the prompt named.

    Honouring the requested signature is part of the API contract under test,
    so a working implementation behind a different name is still a failure --
    a caller written against the prompt would not find it.
    """
    return (
        f"assert hasattr(candidate, {name!r}), "
        f"'candidate does not define {name}'\n"
        f"{name} = getattr(candidate, {name!r})\n"
    )
