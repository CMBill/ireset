"""Compare two regeneration reports for BIT-IDENTICAL behaviour under the
cpu_parallel_optimization_plan comparison contract (section 6).

Everything outside the EXPLICIT exclusion list must be exactly equal:
per-row algorithmic facts (status, t_fail, attempt/check identities, cost
counters, C2/anchor diagnostics, rejection evidence, hand recall, task
stats) and the kept NPZ bytes. Excluded: timing fields, output paths, the
freeze/end/provenance/execution provenance blocks and effective.argv (the
parallel run carries new execution flags).

Exit code 0 = identical within the contract; 1 = any difference (printed
with a JSON-path locator, never a silent pass).

Run: uv run --locked python scripts/cpu_parallel_compare.py \
        --baseline <serial report.json> --candidate <parallel report.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: report-level keys that are pure provenance/timing and never algorithmic
REPORT_EXCLUDED = {"freeze", "end", "provenance", "execution", "notes",
                   "field_notes"}
#: row-level / nested keys pruned recursively (timings, paths, metadata)
KEY_EXCLUDED = {"frame_log", "output_s", "timing_stats"}


def _is_timing_key(key: str) -> bool:
    return (key.endswith("_s") or key.endswith("_ms")
            or key.startswith("ms_per_") or key == "timings_s"
            or key.endswith("_timing"))


def deep_compare(a, b, path: str, diffs: list, *, _prune=frozenset()):
    if isinstance(a, dict) and isinstance(b, dict):
        keys = (set(a) | set(b))
        for k in sorted(keys):
            if k not in a or k not in b:
                diffs.append(f"{path}.{k}: missing in "
                             f"{'baseline' if k not in a else 'candidate'}")
                continue
            if k in _prune or k in KEY_EXCLUDED or _is_timing_key(k):
                continue
            deep_compare(a[k], b[k], f"{path}.{k}", diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: list length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            deep_compare(x, y, f"{path}[{i}]", diffs)
    else:
        if a != b:
            diffs.append(f"{path}: {a!r} != {b!r}")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    base = json.loads(args.baseline.read_text())
    cand = json.loads(args.candidate.read_text())
    diffs: list[str] = []
    # report level minus provenance blocks AND effective (handled below:
    # its argv legitimately carries the new execution flags)
    REPORT_EXCLUDED_EFFECTIVE = REPORT_EXCLUDED | {"effective"}
    deep_compare(base, cand, "report", diffs,
                 _prune=REPORT_EXCLUDED_EFFECTIVE)

    # effective: argv carries the new execution flags; everything else
    # (configs, profile, gate, demos, screening) must be identical
    for k in sorted(set(base.get("effective", {})) | set(cand.get("effective", {}))):
        if k in ("argv",):
            continue
        deep_compare(base["effective"].get(k), cand["effective"].get(k),
                     f"effective.{k}", diffs)

    # kept NPZ bytes (NPZ files are written for KEPT demos, a subset of
    # accepted; the baseline's file set is the reference)
    npz_diffs: list[str] = []
    kept = 0
    for b in sorted(args.baseline.parent.glob("*.npz")):
        kept += 1
        c = args.candidate.parent / b.name
        if not c.is_file():
            npz_diffs.append(f"{b.name} missing in candidate")
            continue
        hb, hc = sha256_file(b), sha256_file(c)
        if hb != hc:
            npz_diffs.append(f"{b.name} sha256 {hb[:16]} != {hc[:16]}")

    ok = not diffs and not npz_diffs
    print(("IDENTICAL" if ok else "MISMATCH")
          + f" rows={len(base.get('rows', []))} kept_npz="
          + str(sum(1 for r in base.get("rows", []) if r.get("accepted"))))
    for d in diffs[:40]:
        print(f"  row-diff {d}")
    for d in npz_diffs:
        print(f"  npz-diff {d}")
    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"ok": ok, "n_diffs": len(diffs) + len(npz_diffs),
             "row_diffs": diffs, "npz_diffs": npz_diffs}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
