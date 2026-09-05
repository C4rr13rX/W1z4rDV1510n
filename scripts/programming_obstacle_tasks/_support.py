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
