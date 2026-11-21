"""Fail-closed outer entry for the one-shot real shared-gradient worker.

The numerical kernel is under implementation, but its production setup and
outer wait/return evidence chain are not integrated or reviewed. This command
therefore remains refusal-only.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ireset.eval import w1_d6_shared_gradient_admission as admission


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    # Intentionally do not read the authorization path, worker inputs, or
    # output root in this implementation-only delivery.
    result = {"schema": admission.SCHEMA + "/supervisor_refusal",
              "status": "refused", "reason": "real_worker_supervisor_integration_not_reviewed",
              "authorization_path_opened": False, "worker_spawned": False,
              "checkpoint_or_batch_reads": 0, "parameter_updates": 0}
    print(json.dumps(result, sort_keys=True))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
