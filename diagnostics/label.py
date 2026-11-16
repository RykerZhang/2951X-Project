"""Failure labeling (owned by the attribution lane). Reads only log records; never imports
the environment code.

label()          T1 rules: uses ground truth (in_I, in_D, ...), i.e. the simulator's view
label_runtime()  T2 rules: may only use fields the agent can observe -- TODO once T2 is defined
"""


def label(records: list[dict]) -> tuple[str, dict]:
    """All records of one chain (in time order) -> (label, extra info).
    For each skill, check in order: start (L3 late / L4 / L1) -> execution (L2) -> end (L3 early).
    The first check that fails is the label of the chain."""
    for i, r in enumerate(records):
        o = r["outcome"]
        if o == "ok":
            continue
        if o in ("stuck", "fail_at_start") and i > 0 \
                and records[i - 1].get("next_ok_if_stopped_at_T_true"):
            return "L3_late", {"pos": i - 1}
        if not r["in_I"]:
            return "L4", {"pos": i, "detected": o == "stuck"}
        if o == "stuck":
            return "L1", {"pos": i}
        if o in ("fail_at_start", "fail_mid"):
            return ("L2" if not r["in_D"] else "other"), {"pos": i}
        if o == "wrong_stop":
            return "L3_early", {"pos": i}
        return "other", {"pos": i}
    return "ok", {}


def label_runtime(records: list[dict]) -> tuple[str, dict]:
    """TODO attribution lane (once T2 is defined): infer the label from fields the agent can see,
    e.g. in_I_learned, outcome, the step at which the failure happened, features.
    Must NOT read ground-truth fields such as in_I, in_D, T_true_at_stop."""
    raise NotImplementedError
