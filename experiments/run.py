"""Execution loop, logging, and batch experiments (owned by the experiments & analysis lane).

Usage:
    python -m experiments.run --model true   --noise 0             # validation condition
    python -m experiments.run --model true   --noise 0.5
    python -m experiments.run --model flawed --noise 0.5
    python -m experiments.run --model tree   --noise 0.5 --n 200   # learned I' and T'
"""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from env.world import Env, features
from skills.library import SKILLS, CHAINS
from learning.models import TrueModel, HandFlawedModel, TreeModel
from diagnostics.label import label

MAX_STEPS = 20


def in_D(env, skill, s) -> bool:
    """D_o: try the policy on a copy of the world and check whether it reaches the subgoal.
    The real world is not affected."""
    env2, s2 = copy.deepcopy(env), copy.deepcopy(s)
    for t in range(MAX_STEPS):
        if skill.subgoal(s2):
            return True
        s2, _ = env2.step(s2, skill.policy(s2, t))
    return skill.subgoal(s2)


def run_skill(env, skill, s, model):
    """Run one skill; return (end state, log record, state when the subgoal was first reached
    or None)."""
    rec = {"skill": skill.name, "features": features(s),
           "in_I": skill.initiation(features(s)),
           "in_I_learned": model.can_start(skill, s),
           "in_D": in_D(env, skill, s)}
    if not rec["in_I_learned"]:
        rec["outcome"] = "stuck"
        return s, rec, None

    first_fail, s_at_true, stopped = None, None, False
    for t in range(MAX_STEPS):
        if s_at_true is None and skill.subgoal(s):
            s_at_true = copy.deepcopy(s)
        if model.is_done(skill, s):
            stopped = True
            break
        s, ok = env.step(s, skill.policy(s, t))
        if not ok and first_fail is None:
            first_fail = t
    if s_at_true is None and skill.subgoal(s):
        s_at_true = copy.deepcopy(s)

    rec["stopped_by_T_learned"] = stopped
    rec["T_true_at_stop"] = skill.subgoal(s)
    if not stopped:
        rec["outcome"] = "fail_at_start" if first_fail == 0 else "fail_mid"
    elif not rec["T_true_at_stop"]:
        rec["outcome"] = "wrong_stop"
    else:
        rec["outcome"] = "ok"
    late = s_at_true is not None and s_at_true.pos != s.pos or \
        (s_at_true is not None and s_at_true.inventory != s.inventory)
    return s, rec, (s_at_true if late else None)


def run_chain(env, chain, s, model):
    records, prev_true_state = [], None
    for name in chain:
        skill = SKILLS[name]
        s, rec, s_true = run_skill(env, skill, s, model)
        rec["next_ok_if_stopped_at_T_true"] = None
        if records and prev_true_state is not None and not rec["in_I"]:
            records[-1]["next_ok_if_stopped_at_T_true"] = \
                skill.initiation(features(prev_true_state))
        records.append(rec)
        prev_true_state = s_true
        if rec["outcome"] != "ok":
            break
    return records


MODELS = {"true": TrueModel, "flawed": HandFlawedModel, "tree": TreeModel}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, default="true")
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--out", default="logs")
    # Knobs of --model tree; see TreeModel in learning/models.py for what they mean
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--source", choices=["chain", "random"], default="chain")
    p.add_argument("--learn", choices=["both", "I", "T"], default="both")
    p.add_argument("--p-start", type=float, default=0.5)
    p.add_argument("--p-done", type=float, default=0.5)
    a = p.parse_args()

    cond = f"{a.model}+noise{a.noise}"
    if a.model == "tree":
        cond = (f"tree-{a.source}-n{a.n}-learn{a.learn}-ps{a.p_start}-pd{a.p_done}"
                f"+noise{a.noise}")
    out = Path(a.out) / f"{cond}.jsonl"
    out.parent.mkdir(exist_ok=True)
    counts = {c: Counter() for c in CHAINS}
    with out.open("w") as fh:
        for seed in range(a.seeds):
            # Trees are trained once per seed (one training set = one learned agent), in worlds
            # with the same randomness as the experiment, and reused for all runs of the seed.
            # The other models are rebuilt for every run, as before.
            tree = TreeModel(seed=seed, n=a.n, source=a.source, learn=a.learn,
                             p_start=a.p_start, p_done=a.p_done, max_steps=MAX_STEPS,
                             layout_noise=a.noise) if a.model == "tree" else None
            for run in range(a.runs):
                for chain_name, chain in CHAINS.items():
                    env = Env(seed=seed * 100_000 + run, layout_noise=a.noise)
                    model = tree or MODELS[a.model](seed=seed * 100_000 + run)
                    records = run_chain(env, chain, env.reset(), model)
                    lab, info = label(records)
                    tag = lab + ("_detected" if info.get("detected") else
                                 "_undetected" if lab == "L4" else "")
                    counts[chain_name][tag] += 1
                    for i, r in enumerate(records):
                        r.update({"condition": cond, "seed": seed, "run": run,
                                  "chain": chain_name, "pos_in_chain": i,
                                  "label_true": lab if info.get("pos") == i else None,
                                  "l4_detected": info.get("detected") if info.get("pos") == i else None,
                                  "label_runtime": None})
                        fh.write(json.dumps(r) + "\n")

    print(f"condition = {cond}   ({a.seeds} seeds x {a.runs} runs per chain)   log -> {out}")
    for chain_name, c in counts.items():
        total = sum(c.values())
        parts = "  ".join(f"{k}={v / total:.0%}" for k, v in sorted(c.items()))
        print(f"  {chain_name:<8} {parts}")


if __name__ == "__main__":
    main()
