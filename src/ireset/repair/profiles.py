"""train24fix3 fixed profile registry (plan sections 4.1/6.1/7.1).

Profiles are the ONLY pre-registered way this round changes solver settings
or C2 row handling.  A profile never edits ``qp_default.yaml``: strict
configs come from the dataclasses, the profile overlay is validated against
a CLOSED field set (unknown keys/inheritance/values are rejected at load
time), and the pair invariants of ``config_io`` still apply after the
overlay.  The registry file itself is content-hashed into every run's
effective record.

Two deliberately separate things:

- ``RepairConfig.osqp_*`` -- the solver settings of every frame WITHOUT
  active C2 rows (and of standard mode).  Their defaults ARE the frozen N0
  behaviour; no profile may change them through this path.
- ``C2Profile.solver`` -- the settings applied ONLY to frames whose C2 rows
  are active (plan 6.1: "新求解设置仅在 regen 且本帧存在启用 C2 行时使用").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import yaml

from ireset.paths import PROJECT_ROOT

PROFILES_YAML = PROJECT_ROOT / "configs" / "repair" / "train24fix3_profiles.yaml"
PROFILE_SCHEMA = "train24fix3_profiles_v1"

#: closed field sets (unknown keys are rejected, never silently dropped)
_SOLVER_FIELDS = {"polish", "eps_abs", "eps_rel", "max_iter", "scaling"}
_PROFILE_FIELDS = {"description", "base", "solver", "c2_mode",
                   "c2_row_precondition", "c2_slack_penalty",
                   "c2_norm_speed_mps", "recovery_policy"}
_C2_MODES = {"axis", "norm_cut"}
_PRECONDITIONS = {"none", "divide_by_bound"}
_SLACK_PENALTIES = {"legacy", "scaled_10000B", "mu"}
#: plan 7.4/7.5: R1 = the feasibility-first recovery phase; None (default)
#: keeps every historical N/S profile on the pure original path
_RECOVERY_POLICIES = {None, "feasibility_first"}
#: the certification entry gate the S bounds must stay strictly below (plan 7.1)
_HAND_ENTRY_SPEED_MPS = 0.05


@dataclass(frozen=True)
class OsqpSettings:
    """One OSQP ``prob.setup`` configuration (N0 defaults = the values
    ``FrameSolver.solve_qp`` historically hard-coded)."""

    polish: bool = False
    eps_abs: float = 3e-5
    eps_rel: float = 3e-5
    max_iter: int = 2500
    scaling: int = 10

    @classmethod
    def from_config(cls, cfg) -> "OsqpSettings":
        """The frame-default settings straight off a RepairConfig."""
        return cls(polish=bool(cfg.osqp_polish), eps_abs=float(cfg.osqp_eps_abs),
                   eps_rel=float(cfg.osqp_eps_rel),
                   max_iter=int(cfg.osqp_max_iter),
                   scaling=int(cfg.osqp_scaling))


@dataclass(frozen=True)
class C2Profile:
    """One pre-registered train24fix3 candidate (N0-N3 / S1-S2)."""

    name: str
    solver: OsqpSettings = field(default_factory=OsqpSettings)
    #: axis = the existing per-axis rows; norm_cut = plan section 7.2 (the
    #: real per-site norm gate + cutting planes; S profiles only)
    c2_mode: str = "axis"
    #: none = original metre-unit C2 QP rows; divide_by_bound = N2/N3's B
    #: preconditioning of the C2 QP rows + dimensionless C2 recovery slack
    c2_row_precondition: str = "none"
    #: legacy = cfg.slack_penalty on metre-unit C2 slack; scaled_10000B =
    #: 10000*B on the dimensionless slack (N2, cost-preserving); mu = the
    #: current merit penalty weight (N3, unit-consistent with merit)
    c2_slack_penalty: str = "legacy"
    #: norm_cut only: the per-site bound V (m/s); None for axis mode
    c2_norm_speed_mps: float | None = None
    #: plan 7.4/7.5 (R candidates): "feasibility_first" activates the
    #: recovery-phase machinery on C2-active regen frames; None = the
    #: original N/S behaviour everywhere (the default of every historical
    #: profile -- the registry version bump keeps their interpretation)
    recovery_policy: str | None = None
    description: str = ""

    def as_record(self) -> dict:
        return asdict(self)


def _parse_settings(section: dict, where: str) -> OsqpSettings:
    unknown = set(section) - _SOLVER_FIELDS
    if unknown:
        raise ValueError(f"{where}: unknown solver keys {sorted(unknown)}")
    s = OsqpSettings(**section)
    if not (s.eps_abs > 0.0 and s.eps_rel > 0.0):
        raise ValueError(f"{where}: eps_abs/eps_rel must be positive")
    if int(s.max_iter) <= 0 or int(s.scaling) <= 0:
        raise ValueError(f"{where}: max_iter/scaling must be positive ints")
    return s


def _parse_profile(name: str, section: dict) -> C2Profile:
    where = f"profile {name!r}"
    if not isinstance(section, dict):
        raise ValueError(f"{where}: mapping required")
    unknown = set(section) - _PROFILE_FIELDS
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    solver = _parse_settings(section.get("solver") or {}, where)
    mode = section.get("c2_mode", "axis")
    if mode not in _C2_MODES:
        raise ValueError(f"{where}: c2_mode must be one of {sorted(_C2_MODES)}")
    precond = section.get("c2_row_precondition", "none")
    if precond not in _PRECONDITIONS:
        raise ValueError(
            f"{where}: c2_row_precondition must be one of {sorted(_PRECONDITIONS)}")
    penalty = section.get("c2_slack_penalty", "legacy")
    if penalty not in _SLACK_PENALTIES:
        raise ValueError(
            f"{where}: c2_slack_penalty must be one of {sorted(_SLACK_PENALTIES)}")
    speed = section.get("c2_norm_speed_mps")
    if mode == "axis":
        if speed is not None:
            raise ValueError(f"{where}: axis mode must not define c2_norm_speed_mps")
    else:
        if not isinstance(speed, (int, float)) or not 0.0 < float(speed):
            raise ValueError(f"{where}: norm_cut requires a positive "
                             f"c2_norm_speed_mps")
        speed = float(speed)
        if speed * (1.0 + 1e-6) >= _HAND_ENTRY_SPEED_MPS:
            raise ValueError(
                f"{where}: c2_norm_speed_mps {speed} does not satisfy "
                f"V*(1+1e-6) < {_HAND_ENTRY_SPEED_MPS} (validator entry gate)")
    recovery = section.get("recovery_policy")
    if recovery not in _RECOVERY_POLICIES:
        raise ValueError(f"{where}: recovery_policy must be one of "
                         f"{sorted(x for x in _RECOVERY_POLICIES if x is not None)}"
                         f" or omitted")
    if recovery is not None and mode != "norm_cut":
        # the pre-registered R candidates are norm-based (plan 7.4); an
        # axis-mode recovery profile is an unregistered combination
        raise ValueError(f"{where}: recovery_policy requires c2_mode "
                         f"norm_cut (the R matrix is norm-based)")
    return C2Profile(name=name, solver=solver, c2_mode=mode,
                     c2_row_precondition=precond, c2_slack_penalty=penalty,
                     c2_norm_speed_mps=speed, recovery_policy=recovery,
                     description=str(section.get("description", "")))


def load_profiles(path=None) -> dict[str, C2Profile]:
    """Parse + validate the registry.  ``base`` inheritance is expanded once
    (unknown/circular bases are rejected); the expanded profile must still
    pass the closed-field validation."""
    p = Path(path) if path else PROFILES_YAML
    doc = yaml.safe_load(p.read_text())
    if not isinstance(doc, dict) or doc.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"{p}: expected schema {PROFILE_SCHEMA!r}")
    raw = doc.get("profiles")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{p}: no profiles registered")
    profiles: dict[str, C2Profile] = {}
    for name, section in raw.items():
        # resolve ``base`` first: an overlay on the BASE's raw mapping, so a
        # derived profile overrides exactly the fields it names
        chain: list[tuple[str, dict]] = []
        cur_name, cur = name, section
        seen = set()
        while cur is not None and "base" in cur:
            base = cur["base"]
            if base not in raw:
                raise ValueError(f"profile {cur_name!r}: unknown base {base!r}")
            if base in seen:
                raise ValueError(f"profile {cur_name!r}: circular base chain")
            seen.add(base)
            chain.append((cur_name, {k: v for k, v in cur.items() if k != "base"}))
            cur_name, cur = base, raw[base]
        merged: dict = dict(cur or {})
        for cname, overlay in reversed(chain):
            merged = {**merged, **overlay, "description":
                      overlay.get("description", merged.get("description", ""))}
        profiles[name] = _parse_profile(name, merged)
    return profiles


def get_profile(name: str, path=None) -> C2Profile:
    profiles = load_profiles(path)
    if name not in profiles:
        raise ValueError(f"unknown train24fix3 profile {name!r}; registered: "
                         f"{sorted(profiles)}")
    return profiles[name]


def profiles_yaml_sha256(path=None) -> str:
    import hashlib

    p = Path(path) if path else PROFILES_YAML
    return hashlib.sha256(p.read_bytes()).hexdigest()
