"""Stage-C comparison: the reviewfix sixseg vs the frozen train24fix2
sixseg and the pre-C2 B sixseg (plan train24fix2_review_fix_plan section
7.C).  Read-only over all inputs; prints a verdict table."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

REGEN = Path("derived/data/teachers_regenerated")
NEW = REGEN / "train24fix2_reviewfix_20261001" / "sixseg"
OLD = REGEN / "train24fix2_20260927_sixseg"
B = REGEN / "train24fix_20260927_sixseg"

COUNT_KEYS = ("attempts", "qp_calls_total", "iterations_total",
              "slack_attempts", "backtracks_adopted", "backtrack_probes",
              "checks_total", "processed_frame_visits",
              "unique_processed_frames")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(tag: str, d: Path) -> dict:
    rep_file = next(d.glob("*_report_diagnosis6_strict.json"))
    rep = json.loads(rep_file.read_text())
    return {"tag": tag, "dir": d, "report": rep,
            "rows": {r["demo"]: r for r in rep["rows"]}}


def counts(row: dict) -> dict:
    return {k: (row.get("cost") or {}).get(k) for k in COUNT_KEYS}


def main() -> None:
    new_rq = load("new_rq", NEW / "rq")
    new_rv = load("new_rv", NEW / "rv")
    new_ns = load("new_ns", NEW / "ns")
    old_rq = load("old_rq", OLD / "rq")
    old_ns = load("old_ns", OLD / "ns")
    b_ns = load("b_ns", B / "ns")

    print("== rq: new vs old (train24fix2 frozen) ==")
    diverged = []
    for demo in sorted(new_rq["rows"]):
        n, o = new_rq["rows"][demo], old_rq["rows"][demo]
        cnt_same = counts(n) == counts(o)
        print(f"  {demo}: accepted new={n['accepted']}/old={o['accepted']} "
              f"t_fail new={n.get('t_fail')}/old={o.get('t_fail')} "
              f"final_stage new={n.get('final_stage')}/old={o.get('final_stage')} "
              f"counts_same={cnt_same}")
        if n["accepted"] != o["accepted"] or n.get("t_fail") != o.get("t_fail") \
                or not cnt_same:
            diverged.append(demo)
        c2n, c2o = n.get("c2_hand_speed") or {}, o.get("c2_hand_speed") or {}
        print(f"    c2 frames_on {c2n.get('frames_on')}/{c2o.get('frames_on')} "
              f"query_miss {len(c2n.get('query_miss', []))}/"
              f"{len(c2o.get('query_miss', []))} "
              f"query_errors {len(c2n.get('query_errors', []))}")
        v1n = n.get("velocity_projection") or {}
        v1o = o.get("velocity_projection") or {}
        print(f"    v1 checks {v1n.get('checks')}/{v1o.get('checks')} "
              f"normalized {v1n.get('normalized')}/{v1o.get('normalized')} "
              f"budget_failures {v1n.get('budget_failures')}/"
              f"{v1o.get('budget_failures')}")
    n_keep = sum(r["accepted"] for r in new_rq["rows"].values())
    o_keep = sum(r["accepted"] for r in old_rq["rows"].values())
    print(f"  keep: new {n_keep}/6 vs old {o_keep}/6  diverged={diverged}")
    if new_rq["rows"] and diverged:
        print("  !! DIVERGENCE on legal paths -- stop and investigate")

    print("== rq vs rv (determinism) ==")
    for demo in sorted(new_rq["rows"]):
        n, v = new_rq["rows"][demo], new_rv["rows"][demo]
        assert n["accepted"] == v["accepted"], demo
        npz_n, npz_v = NEW / "rq" / f"{demo}.npz", NEW / "rv" / f"{demo}.npz"
        if n["accepted"]:
            same = npz_n.exists() and npz_v.exists() and \
                sha256(npz_n) == sha256(npz_v)
            print(f"  {demo}: npz byte-identical={same} "
                  f"counts_same={counts(n) == counts(v)}")
        else:
            print(f"  {demo}: both rejected (t_fail {n.get('t_fail')}/"
                  f"{v.get('t_fail')}) counts_same={counts(n) == counts(v)}")

    print("== ns (standard): new vs old(train24fix2) vs B(pre-C2) ==")
    for demo in sorted(new_ns["rows"]):
        n = new_ns["rows"][demo]
        o = old_ns["rows"].get(demo, {})
        b = b_ns["rows"].get(demo, {})
        print(f"  {demo}: accepted new={n['accepted']} "
              f"(old {o.get('accepted')}, B {b.get('accepted')}) "
              f"t_fail new={n.get('t_fail')}")
        for k in COUNT_KEYS:
            vals = (counts(n).get(k), counts(o).get(k) if o else None,
                    counts(b).get(k) if b else None)
            flag = "" if vals[0] == vals[1] == vals[2] else "   <-- DIFF"
            print(f"    {k}: new={vals[0]} old={vals[1]} B={vals[2]}{flag}")
        v1 = n.get("velocity_projection") or {}
        print(f"    v1 normalized={v1.get('normalized')} "
              f"budget_failures={v1.get('budget_failures')} "
              f"output_modified={v1.get('output_frames_modified')} (expect 0/0/0)")

    print("== drift/config ==")
    for tag, s in (("new_rq", new_rq), ("new_rv", new_rv), ("new_ns", new_ns)):
        e = s["report"]["end"]
        print(f"  {tag}: git_rev={s['report']['git_rev']} "
              f"code_changed_during_run={e['code_changed_during_run']} "
              f"config_unchanged={e['config_unchanged']}")


if __name__ == "__main__":
    main()
