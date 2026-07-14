#!/usr/bin/env python3
"""Teach raw character phrases to reactivate grounded intent neurons."""
from __future__ import annotations

import argparse

ROUTES = [
    ("Build a Python project where operations are refused unless the caller is an administrator, except people may view records they own.",
     "@intent:LANGUAGE:PYTHON\n@intent:SECURITY:AUTHORIZATION\n"),
    ("Produce Python permissions where privileged operators may perform every action and regular people may inspect only their own objects.",
     "@intent:LANGUAGE:PYTHON\n@intent:SECURITY:AUTHORIZATION\n"),
    ("Build a Python order project where repeating the same client command must return the first result without creating another order.",
     "@intent:LANGUAGE:PYTHON\n@intent:API:IDEMPOTENT_COMMAND\n"),
    ("Create Python ordering behavior that remembers a client submission and gives later identical submissions the original outcome.",
     "@intent:LANGUAGE:PYTHON\n@intent:API:IDEMPOTENT_COMMAND\n"),
    ("Build Python balance-movement code where debit and credit either both happen or neither happens when an account is invalid.",
     "@intent:LANGUAGE:PYTHON\n@intent:PERSISTENCE:ATOMIC_TRANSACTION\n"),
    ("Produce Python funds movement that leaves every balance unchanged whenever either side cannot be updated.",
     "@intent:LANGUAGE:PYTHON\n@intent:PERSISTENCE:ATOMIC_TRANSACTION\n"),
    ("Build a Python module emitting machine-readable audit records carrying a request identifier while removing credentials from nested data.",
     "@intent:LANGUAGE:PYTHON\n@intent:OBSERVABILITY:CORRELATED_LOGGING\n@intent:ENTERPRISE:SECRET_REDACTION\n"),
    ("Create Python audit output that attaches a request trace to every record and scrubs credentials at any nesting depth.",
     "@intent:LANGUAGE:PYTHON\n@intent:OBSERVABILITY:CORRELATED_LOGGING\n@intent:ENTERPRISE:SECRET_REDACTION\n"),
    ("Generate Python permissions so administrators can perform anything while ordinary users are limited to viewing objects belonging to them.",
     "@intent:LANGUAGE:PYTHON\n@intent:SECURITY:AUTHORIZATION\n"),
    ("Generate a Python ordering component where duplicate client submissions reuse the original outcome rather than adding a second record.",
     "@intent:LANGUAGE:PYTHON\n@intent:API:IDEMPOTENT_COMMAND\n"),
    ("Write Python funds movement that never leaves only one side changed when validation fails.",
     "@intent:LANGUAGE:PYTHON\n@intent:PERSISTENCE:ATOMIC_TRANSACTION\n"),
    ("Produce Python machine-readable event records carrying a trace value while cleansing login secrets throughout nested payloads.",
     "@intent:LANGUAGE:PYTHON\n@intent:OBSERVABILITY:CORRELATED_LOGGING\n@intent:ENTERPRISE:SECRET_REDACTION\n"),
]


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--endpoint",default="http://127.0.0.1:18600")
    parser.add_argument("--repeats",type=int,default=6); args=parser.parse_args()
    # train_pairs targets pool 4, so perform this route explicitly using its
    # request helpers while keeping raw and semantic frames co-temporal.
    from programming_integrated_retention import b64, request
    for _ in range(args.repeats):
        for prompt, frame in ROUTES:
            request(args.endpoint,"/brain/observe",{"pool_id":1,"frame":b64(prompt)})
            request(args.endpoint,"/brain/observe",{"pool_id":12,"frame":b64(frame)})
            request(args.endpoint,"/brain/tick",{})
    print(f"trained_routes={len(ROUTES)} repeats={args.repeats}")
    return 0


if __name__=="__main__": raise SystemExit(main())
