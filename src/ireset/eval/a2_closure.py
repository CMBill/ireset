"""A2 closure: stop-line registration with states kept separate.

Closure plan ``omniretarget_a2_closure_plan.md`` §3-§4 (C0/C1): re-verify the
existing evidence with the EXISTING comparison entry (never a re-implementation
of the quality gates), classify the known sub12 gap against anything NEW, and
register the U0 stop line while keeping three facts apart that must never be
conflated:

* the ORIGINAL teacher-source verdict (``incomplete``, sole known gap
  ``sub12_largebox_082``) — historical, never rewritten by this round;
* the REGISTRATION status of the closure (whether the stop-line record was
  written from consistent evidence) — a bookkeeping fact, never a source
  comparison pass (§10.1: exit 0 or ``registration_status`` must not be
  readable as "A2 passed");
* the TRAINING readiness (``not_ready`` until a future plan says otherwise).

All judgement functions are pure (dicts in, dicts out) so the counterexamples
of plan §10.1 (tampered report, shrunken denominator, dropped rows, a
registration masquerading as a verdict) are unit-testable without the real
artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

#: the sole demo the baseline evidence gap is allowed to mention (§3.1)
KNOWN_MISSING_DEFAULT = ("sub12_largebox_082",)

#: report fields that legitimately differ between two comparison runs of the
#: same inputs (§3.1 item 6): wall-clock facts and the isolated attempt id —
#: everything else must match bit for bit
RUN_SCOPED_FIELDS = ("timestamp", "comparison_attempt", "cost_s")

_VERDICTS = {"local_preferred", "no_quality_gain", "incomplete", "not_ready"}


def _diff_paths(a, b, prefix: str = "") -> list[str]:
    """JSON paths at which two report subtrees differ (order-sensitive)."""
    def join(key: str) -> str:
        return f"{prefix}.{key}" if prefix else str(key)
    if type(a) is not type(b):
        return [prefix or "<root>"]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(join(k))
            else:
                out += _diff_paths(a[k], b[k], join(k))
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [prefix or "<root>"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diff_paths(x, y, f"{prefix}[{i}]")
        return out
    if isinstance(a, float) and isinstance(b, float):
        # invalid-row sentinels serialize NaN on both sides; two NaNs from
        # bit-identical reports are EQUAL (NaN != NaN would flag every
        # missing-demo row as a difference)
        import math
        same = (a == b) or (math.isnan(a) and math.isnan(b))
        return [] if same else [prefix or "<root>"]
    return [] if a == b else [prefix or "<root>"]


def verify_plan_anchors(paths: dict[str, Path], declared: dict[str, str],
                        hash_file) -> dict:
    """Measured SHA-256 of the plan's §1.1 anchor files vs the declared ones.

    The plan's anchor table pins the bytes the closure is allowed to reason
    over: a freeze directory that still exists is not evidence until its
    current bytes match the frozen hashes (a manifest file being unchanged
    does not mean the files it lists are unchanged).
    """
    out: dict[str, dict] = {}
    for key, declared_sha in declared.items():
        path = paths[key]
        rec: dict = {"path": str(path), "declared_sha256": declared_sha}
        if not path.is_file():
            rec.update(measured_sha256=None, matches=False,
                       reason="missing")
        else:
            measured = hash_file(path)
            rec.update(measured_sha256=measured, matches=measured == declared_sha)
        out[key] = rec
    return {"anchors": out,
            "passed": all(r["matches"] for r in out.values())}


def compare_fresh_to_authoritative(authoritative: dict, fresh: dict) -> dict:
    """Numeric comparison of a fresh comparison report vs the authoritative one.

    Only the run-scoped fields (``RUN_SCOPED_FIELDS``) are excluded; the
    metrics, every gate, the evidence gaps, the admission blocks (input
    re-hashes, asset re-hashes, env triple, artifact chain, bindings, actual
    consumed data paths) must be identical.
    """
    a = {k: v for k, v in authoritative.items() if k not in RUN_SCOPED_FIELDS}
    f = {k: v for k, v in fresh.items() if k not in RUN_SCOPED_FIELDS}
    diffs = _diff_paths(a, f)
    return {"matches": not diffs, "differences": diffs,
            "excluded_fields": list(RUN_SCOPED_FIELDS)}


def split_gaps(evidence_gaps: list[str],
               known_missing=KNOWN_MISSING_DEFAULT) -> dict:
    """Known gaps (the registered sub12 evidence gap) vs anything new.

    A gap is KNOWN iff it mentions a known-missing demo — the baseline's
    ``sub12_largebox_082:{U0_invalid,binding_not_verified}`` rows and the
    matching ``admission:`` reason. Any other gap (a different demo, an
    input/asset/env/chain/binding failure) is an ``unexpected_evidence_gap``
    and must stop the closure before a stop line is registered (§3.1).
    """
    known = [g for g in evidence_gaps if any(d in g for d in known_missing)]
    unexpected = [g for g in evidence_gaps if g not in known]
    return {"known": known, "unexpected": unexpected}


def check_report_invariants(report: dict, expected_pilot: list[str],
                            u_side: str = "U0") -> dict:
    """Structural invariants of a comparison report used as closure evidence.

    Rejects the §10.1 counterexamples: a shrunken denominator (pilot trimmed
    to the five measurable demos), a dropped row, a verdict word outside the
    frozen vocabulary, or an evidence/verdict disagreement. ``u_side`` names
    the generated-source metrics key ("U0" for baseline comparisons, "UR"
    for recovery reports merged at C2).
    """
    violations: list[str] = []
    pilot = report.get("pilot")
    if pilot != expected_pilot:
        violations.append(
            f"pilot_mismatch:{pilot}!={expected_pilot}")
    metrics = report.get("metrics", {})
    for side in ("P", u_side):
        rows = metrics.get(side, {})
        if sorted(rows) != sorted(expected_pilot):
            violations.append(f"rows_missing_or_extra:{side}:"
                              f"{sorted(rows)}!={sorted(expected_pilot)}")
    verdict = report.get("verdict")
    if verdict not in _VERDICTS:
        violations.append(f"verdict_out_of_vocabulary:{verdict}")
    vr = report.get("verdict_record", {})
    gaps = vr.get("evidence_gaps", [])
    if bool(vr.get("complete_evidence")) != (not gaps):
        violations.append("complete_evidence_disagrees_with_gaps")
    if verdict == "local_preferred" and not vr.get("complete_evidence"):
        violations.append("local_preferred_without_complete_evidence")
    binding = report.get("binding", {})
    if sorted(binding) != sorted(expected_pilot):
        violations.append(f"binding_keys_mismatch:{sorted(binding)}")
    return {"violations": violations, "passed": not violations}


def check_consumption(attempts_index: list[dict], coverage: dict,
                      pilot: list[str], known_missing=KNOWN_MISSING_DEFAULT,
                      attempt_scopes: dict[str, list[str]] | None = None,
                      manifest_rows: dict[str, list[dict]] | None = None,
                      ) -> dict:
    """Current U0 consumption map vs the facts the baseline registered (§3.1
    item 5): the five measurable demos still consume ``attempt_001``, the
    known-missing demo has no OK artifact in ANY attempt, and the two
    historical failed attempts are still retained (evidence is never
    garbage-collected into a shrunken denominator).

    Review fix (2026-10-10): retention is verified on the ROWS, not just on
    the attempt-directory names — each retained attempt's manifest must still
    carry a row for every demo its index entry declares in scope, so deleting
    one of sub12's two registered failure rows shrinks the stop line's cost
    basis and is caught here (``failure_record_dropped``).
    """
    violations: list[str] = []
    names = [a.get("attempt") for a in attempts_index]
    if len(names) != len(set(names)):
        violations.append("duplicate_attempt_names")
    by_demo = coverage.get("by_demo", {})
    missing = coverage.get("missing", [])
    if missing != list(known_missing):
        violations.append(f"missing_set_changed:{missing}")
    expected_consumers = [d for d in pilot if d not in known_missing]
    for demo in expected_consumers:
        if by_demo.get(demo, {}).get("attempt") != "attempt_001":
            violations.append(f"consumption_changed:{demo}:"
                              f"{by_demo.get(demo, {}).get('attempt')}")
    if attempt_scopes is not None and manifest_rows is not None:
        for att, scope in sorted(attempt_scopes.items()):
            rows = manifest_rows.get(att)
            if rows is None:
                violations.append(f"manifest_rows_unavailable:{att}")
                continue
            have = {r.get("demo") for r in rows}
            dropped = sorted(set(scope) - have)
            if dropped:
                violations.append(f"failure_record_dropped:{att}:{dropped}")
        # the stop line's cost basis: every known-missing demo must still
        # carry at least one registered failure row somewhere
        for demo in known_missing:
            n_failed = sum(
                1 for rows in manifest_rows.values() for r in rows
                if r.get("demo") == demo and r.get("status") != "ok")
            if n_failed < 1:
                violations.append(f"cost_basis_empty:{demo}")
    return {"violations": violations, "passed": not violations,
            "attempts_retained": names,
            "consumed_from_attempt_001": sorted(
                d for d, e in by_demo.items() if e.get("attempt") == "attempt_001"),
            "missing": missing}


def summarize_quality_evidence(report: dict) -> dict:
    """g2-g5 per-demo evidence extracted from the baseline report (§4).

    Five-demo results are reported SEPARATELY from the six-demo formal
    verdict: the formal verdict stays ``incomplete`` because of sub12; the
    five measurable demos are the stop-line's quality basis.
    """
    vr = report["verdict_record"]
    gates = vr["gates"]
    pilot = report["pilot"]
    measurable = [d for d in pilot
                  if report["metrics"]["U0"][d].get("valid")]
    return {
        "formal_verdict": report["verdict"],
        "formal_verdict_note": "six-demo verdict; incomplete solely for the "
                               "known sub12 evidence gap",
        "measurable_demos": measurable,
        "g2_task_in_bound": {d: gates["g2_task_mpjpe"]["per_demo"][d]
                             for d in measurable},
        "g3_per_demo": {d: gates["g3_penetration_and_feet"]["per_demo"][d]
                        for d in measurable},
        "g4": {k: gates["g4_hand_improvement"][k]
               for k in ("primary_metric", "n_coverage_gain_demos",
                         "n_gap_drop_demos", "passed")},
        "g5_per_demo": {d: gates["g5_degradation"]["per_demo"][d]
                        for d in measurable},
        "gates_passed": {k: v["passed"] for k, v in gates.items()},
    }


def _training_unmet(recovery_status: str | None) -> list[str]:
    """Why training stays ``not_ready`` — the reasons must MATCH the merged
    recovery state (review fix 13, 2026-10-10): once a complete six-demo
    ``no_quality_gain`` exists, "no source has produced a complete verdict"
    is false; what actually gates training is the absence of a
    ``local_preferred`` source and of the W1 A3/B/C prerequisite plan.
    """
    if recovery_status == "no_quality_gain":
        return [
            "no teacher source has reached local_preferred; the complete "
            "no_quality_gain verdict (UR recovery) authorizes at most the "
            "W1 weak-supervision diagnostic, never train24 labels "
            "(parent plan §3.4)",
            "W1 A3/B/C weak-supervision diagnosis requires its own "
            "preregistered plan and has not been started (parent §3.6)",
            "train24/T1 small-batch entry requires its own plan and "
            "prerequisites (parent plan §3.4-§3.6)",
        ]
    return [
        "formal A2 verdict is incomplete (known gap: no valid U0 "
        "artifact for sub12_largebox_082)",
        "no source experiment has produced a complete six-demo "
        "local_preferred or no_quality_gain verdict",
        "W1/T1/train24 small-batch entry requires its own plan and "
        "prerequisites (parent plan §3.4-§3.6)",
    ]


def build_closure_record(
    *,
    baseline: dict,
    verification: dict,
    quality_evidence: dict,
    cost_basis: dict,
    diagnosis: dict | None = None,
    recovery: dict | None = None,
) -> dict:
    """Assemble the C1 stop-line record (§4); statuses stay separated.

    ``registration_status`` says whether THIS registration completed from
    consistent evidence — it is not, and must not be usable as, a teacher
    source verdict. The training block is pinned to ``not_ready`` here: this
    round registers a stop line, it never opens a training entry.

    Review fix 7 (2026-10-10): the registration inherits EVERY C0 failure
    condition — an unstable repo identity during the run or a failed
    upstream-checkout check blocks it exactly like an anchor mismatch (the
    CLI passes both legs into ``verification``).
    """
    unexpected = verification.get("unexpected_gaps") or []
    violations = (verification.get("report_violations")
                  or []) + (verification.get("consumption_violations") or [])
    identity_stable = verification.get("identity_stable", False)
    checkout_passed = verification.get("checkout_passed", False)
    ok = (not unexpected and not violations
          and verification.get("anchors_passed", False)
          and verification.get("fresh_matches", False)
          and identity_stable and checkout_passed)
    recovery_rec = recovery or {"status": "not_registered"}
    record = {
        "stage": "C1 U0 stop-line registration (closure plan §4)",
        "baseline": baseline,
        "verification": verification,
        "disposition": {
            "action": "stop_u0_expansion",
            "meaning": "no further generation attempts under the original U0 "
                       "configuration; the recorded attempts stay as evidence",
            "quality_basis": quality_evidence,
            "cost_basis": cost_basis,
        },
        "quality_evidence": quality_evidence,
        "diagnosis": diagnosis or {"status": "not_run"},
        "recovery": recovery_rec,
        "training": {
            "status": "not_ready",
            "unmet": _training_unmet(recovery_rec.get("status")),
        },
        "baseline_verdict_unchanged": True,
        "registration_status": {
            "complete": bool(ok),
            "blocking_reasons": (["unexpected_evidence_gap"] if unexpected else [])
            + (violations if violations else [])
            + ([] if verification.get("anchors_passed", False) else ["plan_anchor_mismatch"])
            + ([] if verification.get("fresh_matches", False) else ["fresh_comparison_differs"])
            + ([] if identity_stable else ["identity_unstable_during_run"])
            + ([] if checkout_passed else ["upstream_checkout_mismatch"]),
        },
    }
    return record


def render_markdown(record: dict) -> str:
    """Human-facing stop-line summary; must show the two pinned statuses."""
    b = record["baseline"]
    v = record["verification"]
    d = record["disposition"]
    q = record["quality_evidence"]
    lines = [
        "# A2 收口停线登记（U0）",
        "",
        f"- 基线报告：`{b['report_path']}`",
        f"- 基线 SHA-256：`{b['report_sha256']}`",
        f"- **baseline_verdict = {b['verdict']}**（唯一已知缺口："
        f"{', '.join(b['missing'])}）",
        f"- 可测演示（{len(q['measurable_demos'])}/6）："
        f"{', '.join(q['measurable_demos'])}",
        "",
        "## 复验（C0）",
        f"- 方案锚点一致：{v['anchors_passed']}",
        f"- 新鲜比较与权威报告逐位一致（排除 {', '.join(RUN_SCOPED_FIELDS)}）："
        f"{v['fresh_matches']}",
        f"- 已知缺口：{v['known_gaps'] or '（无）'}",
        f"- 新增缺口：{v['unexpected_gaps'] or '（无）'}",
        f"- 消费映射：五段 attempt_001 保留；历史失败 attempts "
        f"{v.get('attempts_retained', [])} 均在",
        "",
        "## 停线处置",
        f"- action = {d['action']}",
        f"- 质量依据：g3/g5 五段全败、g4 改善段数 "
        f"{q['g4']['n_gap_drop_demos']}/3 不达标、g2 五段在界",
        f"- 成本依据：sub12 默认配置确定性不可行"
        f"（{json.dumps(d['cost_basis'], ensure_ascii=False)}）",
        "",
        "## 状态",
        f"- **training = {record['training']['status']}**",
        f"- training 未满足门：{'；'.join(record['training']['unmet'])}",
        f"- diagnosis = {record['diagnosis']['status']}",
        f"- recovery = {record['recovery']['status']}",
        f"- registration_status.complete = "
        f"{record['registration_status']['complete']}",
        "",
        "登记状态不是来源比较通过；原 `teacher_selection.json` 的裁决与哈希"
        "保持不变。任何恢复实验须按方案 §6 事前注册。",
        "",
    ]
    return "\n".join(lines)
