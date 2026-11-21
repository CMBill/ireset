#!/usr/bin/env python3
"""Synthetic-only I/C acceptance for the W1 shared-gradient draft."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ireset.eval import w1_d6_shared_gradient as shared


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--cpu-synthetic", action="store_true",
                       help="run hand-computable CPU controls; no model or source inputs")
    modes.add_argument("--cuda-synthetic", action="store_true",
                       help="run the nine-by-two production-architecture synthetic acceptance")
    modes.add_argument("--real", action="store_true",
                       help="reserved real entry; always refuses in this implementation")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--authorization", type=Path,
                        help="reserved binding; never opened by the current fail-closed entry")
    args = parser.parse_args()
    try:
        if args.real:
            shared.real_entry(args.out, authorization=None)
        elif args.cpu_synthetic:
            if args.out is not None:
                raise shared.SharedGradientError("cpu_synthetic_output_not_supported")
            result = shared.run_cpu_synthetic_acceptance()
            print(json.dumps(result, sort_keys=True, allow_nan=False))
        else:
            if args.authorization is not None:
                raise shared.SharedGradientError("synthetic_mode_rejects_authorization_path")
            if args.out is None:
                raise shared.SharedGradientError("cuda_synthetic_requires_exclusive_output_path")
            result = shared.run_cuda_synthetic_acceptance(args.out)
            print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except shared.SharedGradientError as exc:
        print(json.dumps({"schema": shared.SCHEMA, "status": "refused", "reason": str(exc)},
                         sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
