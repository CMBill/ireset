"""Fail-closed outer entry for the future one-shot real shared-gradient worker.

This delivery freezes the launch boundary but does not provide or launch the
numerical worker. A later revision must bind and review that worker before this
command can be enabled.
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
              "status": "refused", "reason": "real_worker_not_implemented_and_reviewed",
              "authorization_path_opened": False, "worker_spawned": False,
              "checkpoint_or_batch_reads": 0, "parameter_updates": 0}
    print(json.dumps(result, sort_keys=True))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
