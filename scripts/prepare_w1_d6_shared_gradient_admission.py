"""Create an unsigned, metadata-only real-execution admission draft."""

import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ireset.eval import w1_d6_shared_gradient_admission as admission


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--write-draft", action="store_true", required=True)
    parser.add_argument("--final-review", type=Path)
    args = parser.parse_args(argv)
    if args.final_review is not None:
        package, draft = admission.build_reviewed_materials(ROOT, args.final_review)
        ledger_root = ROOT / admission.AUTHORIZATION_LEDGER_REL
        ledger_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    else:
        package, draft = admission.build_materials(ROOT)
    parent = ROOT / admission.REVIEW_DIR
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    existing = [int(path.name.removeprefix("attempt_")) for path in parent.glob("attempt_*")
                if path.is_dir() and re.fullmatch(r"attempt_[0-9]{3}", path.name)]
    out = parent / f"attempt_{max(existing, default=0) + 1:03d}"
    out.mkdir(mode=0o700, exist_ok=False)
    fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    for name, doc in (("admission.json", package), ("authorization_draft.json", draft)):
        path = out / name
        raw = (json.dumps(doc, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({"state": package["state"], "authorized": False,
                      "checkpoint_or_batch_reads": 0,
                      "output": str(out.relative_to(ROOT))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
