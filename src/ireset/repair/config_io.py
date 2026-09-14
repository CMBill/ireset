"""Paired strict/relaxed mode loading for repair + validation configs.

P0 plan section 4 (L2): a mode must drive BOTH configs together -- relaxing
only the final validator is a proven no-op for ``rejected_frame`` failures,
and repair-side bands above validator gates make the repairer produce output
the final check then rejects. ``check_pair_invariants`` enforces
"repair band <= validator gate" at load time (previously comment-only).

``strict`` is the dataclass default; ``relaxed`` overlays the
``relaxed_repair`` / ``relaxed_validation`` yaml sections (census-derived
teacher-baseline values, never valid for plan 10.1 claims).
"""

from __future__ import annotations

from dataclasses import fields, replace

import yaml

from ireset.paths import PROJECT_ROOT
from ireset.repair.context import RepairConfig
from ireset.repair.validation import ValidationConfig

MODES = ("strict", "relaxed")
_QP_YAML = PROJECT_ROOT / "configs" / "repair" / "qp_default.yaml"
_PROTOCOL_YAML = PROJECT_ROOT / "configs" / "eval" / "protocol.yaml"


def _overlay(cfg, section: dict, src: str):
    names = {f.name for f in fields(cfg)}
    unknown = set(section) - names
    if unknown:
        raise ValueError(f"unknown {src} keys: {sorted(unknown)}")
    return replace(cfg, **section)


def load_mode_configs(mode: str) -> tuple[RepairConfig, ValidationConfig]:
    if mode == "strict":
        rcfg, vcfg = RepairConfig(), ValidationConfig()
    elif mode == "relaxed":
        qp = yaml.safe_load(_QP_YAML.read_text())["relaxed_repair"]
        pr = yaml.safe_load(_PROTOCOL_YAML.read_text())["relaxed_validation"]
        rcfg = _overlay(RepairConfig(), qp, "relaxed_repair")
        vcfg = _overlay(ValidationConfig(), pr, "relaxed_validation")
    else:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    check_pair_invariants(rcfg, vcfg)
    return rcfg, vcfg


def check_pair_invariants(rcfg: RepairConfig, vcfg: ValidationConfig) -> None:
    """Repair-side bands must not exceed the validator gates they pair with."""
    pairs = [
        ("foot_contact_gap_m", rcfg.foot_contact_gap_m, "foot_ground_gap_m", vcfg.foot_ground_gap_m),
        ("contact_band_gap_m", rcfg.contact_band_gap_m, "hand_gap_m", vcfg.hand_gap_m),
        ("hand_anchor_tol_m", rcfg.hand_anchor_tol_m, "hand_anchor_error_m", vcfg.hand_anchor_error_m),
        ("foot_anchor_tol_m", rcfg.foot_anchor_tol_m, "foot_anchor_drift_m", vcfg.foot_anchor_drift_m),
        ("foot_speed_mps", rcfg.foot_speed_mps, "foot_speed_mps", vcfg.foot_speed_mps),
        # penetration: both are magnitudes of tolerated penetration; the
        # repairer must never tolerate more than the final check accepts
        ("contact_band_pen_m", rcfg.contact_band_pen_m,
         "penetration_tolerance_m", vcfg.penetration_tolerance_m),
    ]
    errs = [f"repair {rn}={rv} > validator {vn}={vv}" for rn, rv, vn, vv in pairs if rv > vv + 1e-12]
    if errs:
        raise ValueError("repair/validation pair invariant violated: " + "; ".join(errs))
