# constraint_layer/sti_rules.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Data structures
# ============================================================

@dataclass
class RuleTrigger:
    """Structured audit record for transparency & thesis-grade traceability."""
    rule_id: str
    severity: str  # "info" | "warning" | "high" | "critical"
    message: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    changed: Dict[str, Any] = field(default_factory=dict)


@dataclass
class STIConstraints:
    """
    Safety envelope / deterministic constraints for tunnel control.

    These limits can later be loaded from config (yaml) or tuned per tunnel.
    """
    # Ventilation
    max_fan_stage: int = 3
    max_fan_step_per_s: int = 1  # ramp rate in "stage per second"

    # VMS
    min_vms_speed_kmh: int = 60
    max_vms_speed_kmh: int = 100

    # Incident behavior
    incident_no_speed_increase: bool = True

    # Optional: if incident active, enforce max fan stage at least (e.g. smoke extraction)
    incident_min_fan_stage: int = 0


# ============================================================
# Helpers: tag picking & derived states
# ============================================================

def _get_float(tags: Dict[str, Any], key: str, default: float = 0.0) -> float:
    v = tags.get(key, default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _any_true(tags: Dict[str, Any], keys: List[str], threshold: float = 0.5) -> bool:
    for k in keys:
        if _get_float(tags, k, 0.0) >= threshold:
            return True
    return False


def derive_state(current_tags: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derive higher-level flags from the raw tag snapshot.
    This is intentionally conservative and robust if some tags are missing.

    Returns a dict with Z3-like booleans (can be used in UI or logging).
    """
    out: Dict[str, Any] = {}

    # Incident active: support both your earlier placeholder and your tags.yaml naming
    incident = _any_true(
        current_tags,
        keys=[
            "Z2.EVENT.IncidentFlag",        # legacy / example
            "Z3.EVT.Incident.Active",       # tags.yaml
        ],
        threshold=0.5,
    )
    out["incident_active"] = bool(incident)

    # Jam active: support your Z3 event label
    jam = _any_true(
        current_tags,
        keys=[
            "Z3.EVT.Jam.Active",
            "Z3.TRAF.STATE.Tunnel.JamDetected",
        ],
        threshold=0.5,
    )
    out["jam_active"] = bool(jam)

    # CO alarm/warn (if present)
    co_warn = _any_true(current_tags, ["Z3.ENV.ALARM.CO.Warn"], 0.5)
    co_alarm = _any_true(current_tags, ["Z3.ENV.ALARM.CO.Alarm"], 0.5)
    vis_alarm = _any_true(current_tags, ["Z3.ENV.ALARM.Visibility.Alarm"], 0.5)

    out["co_warn"] = bool(co_warn)
    out["co_alarm"] = bool(co_alarm)
    out["vis_alarm"] = bool(vis_alarm)

    return out


# ============================================================
# Core: stateless filter (backward compatible)
# ============================================================

def filter_recommendation(
    rec: Dict[str, Any],
    current_tags: Dict[str, Any],
    prev_rec: Optional[Dict[str, Any]] = None,
    c: STIConstraints = STIConstraints(),
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Backward compatible API:
    Returns (filtered_recommendation, audit_info)

    rec: {"fan_stage": int, "vms_speed": int}
    audit_info: {"violations": [..], "changed": {...}, "triggers": [RuleTrigger-like dicts]}
    """
    triggers: List[RuleTrigger] = []
    audit = {"violations": [], "changed": {}, "triggers": []}
    out = dict(rec)

    state = derive_state(current_tags)
    incident = state["incident_active"]

    # ------------------------------------------------------------
    # R1: Incident => no speed increase
    # ------------------------------------------------------------
    if incident and c.incident_no_speed_increase and "vms_speed" in out:
        # Support either legacy speed limit tag or your VMS setpoint tag
        current_limit = None
        if "Z2.VMS.SpeedLimit" in current_tags:
            current_limit = int(_get_float(current_tags, "Z2.VMS.SpeedLimit", 80))
        elif "Z1.VMS.SIGN.V01.SpeedSet" in current_tags:
            current_limit = int(_get_float(current_tags, "Z1.VMS.SIGN.V01.SpeedSet", 80))
        else:
            current_limit = 80

        proposed = int(out["vms_speed"])
        if proposed > current_limit:
            audit["violations"].append("INCIDENT_NO_SPEED_INCREASE")
            audit["changed"]["vms_speed_incident"] = (proposed, current_limit)
            out["vms_speed"] = current_limit
            triggers.append(
                RuleTrigger(
                    rule_id="R_INCIDENT_NO_SPEED_INCREASE",
                    severity="critical",
                    message="During incident: speed increase is not allowed.",
                    evidence={"proposed_vms_speed": proposed, "current_speed_limit": current_limit},
                    changed={"vms_speed": (proposed, current_limit)},
                )
            )

    # ------------------------------------------------------------
    # R2: Clamp VMS band
    # ------------------------------------------------------------
    if "vms_speed" in out:
        v = int(out["vms_speed"])
        v2 = max(c.min_vms_speed_kmh, min(c.max_vms_speed_kmh, v))
        if v2 != v:
            audit["violations"].append("VMS_CLAMP")
            audit["changed"]["vms_speed"] = (v, v2)
            out["vms_speed"] = v2
            triggers.append(
                RuleTrigger(
                    rule_id="R_VMS_CLAMP",
                    severity="high",
                    message="VMS speed setpoint clamped to allowed band.",
                    evidence={"proposed_vms_speed": v, "min": c.min_vms_speed_kmh, "max": c.max_vms_speed_kmh},
                    changed={"vms_speed": (v, v2)},
                )
            )

    # ------------------------------------------------------------
    # R3: Clamp fan stage + optional incident min fan stage
    # ------------------------------------------------------------
    if "fan_stage" in out:
        f = int(out["fan_stage"])

        # optional: enforce min fan stage during incident (e.g., safety ventilation)
        if incident and c.incident_min_fan_stage > 0 and f < c.incident_min_fan_stage:
            audit["violations"].append("INCIDENT_MIN_FAN_STAGE")
            audit["changed"]["fan_stage_incident_min"] = (f, c.incident_min_fan_stage)
            f = c.incident_min_fan_stage
            triggers.append(
                RuleTrigger(
                    rule_id="R_INCIDENT_MIN_FAN_STAGE",
                    severity="high",
                    message="During incident: minimum fan stage enforced.",
                    evidence={"proposed_fan_stage": int(out["fan_stage"]), "incident_min_fan_stage": c.incident_min_fan_stage},
                    changed={"fan_stage": (int(out["fan_stage"]), c.incident_min_fan_stage)},
                )
            )

        f2 = max(0, min(c.max_fan_stage, f))
        if f2 != f:
            audit["violations"].append("FAN_CLAMP")
            audit["changed"]["fan_stage"] = (f, f2)
            out["fan_stage"] = f2
            triggers.append(
                RuleTrigger(
                    rule_id="R_FAN_CLAMP",
                    severity="high",
                    message="Fan stage clamped to allowed maximum.",
                    evidence={"proposed_fan_stage": f, "max_fan_stage": c.max_fan_stage},
                    changed={"fan_stage": (f, f2)},
                )
            )
        else:
            out["fan_stage"] = f2

        # ------------------------------------------------------------
        # R4: Ramp rate limit fan stage change
        # ------------------------------------------------------------
        if prev_rec and "fan_stage" in prev_rec:
            prev_f = int(prev_rec["fan_stage"])
            if abs(out["fan_stage"] - prev_f) > c.max_fan_step_per_s:
                audit["violations"].append("FAN_RAMP_RATE")
                limited = prev_f + c.max_fan_step_per_s * (1 if out["fan_stage"] > prev_f else -1)
                audit["changed"]["fan_stage_ramp"] = (out["fan_stage"], limited)
                out["fan_stage"] = limited
                triggers.append(
                    RuleTrigger(
                        rule_id="R_FAN_RAMP_RATE",
                        severity="warning",
                        message="Fan ramp rate limited (stage/s).",
                        evidence={"prev_fan_stage": prev_f, "proposed_after_clamp": int(prev_rec.get("fan_stage", prev_f)), "max_step_per_s": c.max_fan_step_per_s},
                        changed={"fan_stage": (audit["changed"]["fan_stage_ramp"][0], limited)},
                    )
                )

    # Attach triggers to audit (as dicts for JSON-friendly UI)
    audit["triggers"] = [
        {
            "rule_id": t.rule_id,
            "severity": t.severity,
            "message": t.message,
            "evidence": t.evidence,
            "changed": t.changed,
        }
        for t in triggers
    ]

    return out, audit


# ============================================================
# Stateful engine (recommended for streaming/playback)
# ============================================================

@dataclass
class STIEngine:
    """
    Stateful STI wrapper that remembers the previous filtered recommendation,
    so ramp-rate constraints work correctly across timesteps.
    """
    constraints: STIConstraints = field(default_factory=STIConstraints)
    prev_filtered: Optional[Dict[str, Any]] = None

    def step(self, rec: Dict[str, Any], current_tags: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        filtered, audit = filter_recommendation(
            rec=rec,
            current_tags=current_tags,
            prev_rec=self.prev_filtered,
            c=self.constraints,
        )
        self.prev_filtered = dict(filtered)
        return filtered, audit

    def reset(self) -> None:
        self.prev_filtered = None