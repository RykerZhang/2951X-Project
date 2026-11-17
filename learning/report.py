"""Per-skill report for the learned I' and T' (owned by the learning lane).

    python -m learning.report --n 200 --noise 0.5 --source chain

Trains one TreeModel, prints the learned rules, and scores them on held-out real runs from BOTH
start-state distributions ("chain" and "random"), whichever one the model was trained on. A
model can look accurate on random states and still be wrong on the states the chain hands to
it, so always read both. The held-out worlds are never used for training. The same seed and
knobs give exactly the trees that experiments.run trains for that seed.

What the error columns mean for the chain (fractions of all held-out examples, so
acc + the two error columns = 1):
    false_accept   I' says "can start" but the real run fails     -> L2 or undetected L4
    false_reject   I' says "cannot start" but the run would work  -> L1
    false_done     T' says "done" but the subgoal is not reached  -> L3 early
    missed_done    the subgoal is reached but T' says "not done"  -> L3 late
"""
import argparse

from learning.data import collect
from learning.models import TreeModel

ERRORS = {"I": ("false_accept", "false_reject"), "T": ("false_done", "missed_done")}


def evaluate(model, data) -> dict:
    """Score the trees of `model` on `data` (the output of learning.data.collect).
    The truth is the label in the data: the outcome of the real run (I') or skill.subgoal (T').
    Returns {skill: {"I": {"n", "acc", "false_accept", "false_reject"},
                     "T": {"n", "acc", "false_done", "missed_done"}}};
    a skill/kind without a tree or without data is left out."""
    threshold = {"I": model.p_start, "T": model.p_done}
    out = {}
    for name, d in data.items():
        for kind, (false_yes, false_no) in ERRORS.items():
            rows = d[kind]
            if not rows or name not in model.trees[kind]:
                continue
            says = [model.prob(kind, name, x) >= threshold[kind] for x, _ in rows]
            n = len(rows)
            out.setdefault(name, {})[kind] = {
                "n": n,
                "acc": sum(yes == bool(y) for yes, (_, y) in zip(says, rows)) / n,
                false_yes: sum(yes and not y for yes, (_, y) in zip(says, rows)) / n,
                false_no: sum(not yes and bool(y) for yes, (_, y) in zip(says, rows)) / n,
            }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--source", choices=["chain", "random"], default="chain")
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--p-start", type=float, default=0.5)
    p.add_argument("--p-done", type=float, default=0.5)
    p.add_argument("--heldout", type=int, default=500, help="held-out runs per skill")
    a = p.parse_args()

    model = TreeModel(seed=a.seed, n=a.n, source=a.source, p_start=a.p_start, p_done=a.p_done,
                      layout_noise=a.noise)
    scores = {src: evaluate(model, collect(src, a.heldout, a.seed, split="heldout",
                                           layout_noise=a.noise))
              for src in ("chain", "random")}

    print(f"TreeModel(seed={a.seed}, n={a.n}, source={a.source}, p_start={a.p_start}, "
          f"p_done={a.p_done}), worlds with layout_noise={a.noise}")
    for name, d in model.data.items():
        print(f"\n===== {name} =====")
        for kind, what in (("I", "runs succeeded"), ("T", "states with the subgoal reached")):
            rows = d[kind]
            if not rows:
                print(f"{kind}': no training data")
                continue
            share = sum(y for _, y in rows) / len(rows)
            print(f"{kind}' rules  (trained on {len(rows)} examples, {share:.0%} {what})")
            print(model.rules(kind, name), end="")
            for src in ("chain", "random"):
                m = scores[src].get(name, {}).get(kind)
                if m:
                    print(f"  held-out {src:<6} " +
                          "  ".join(f"{k}={v}" if k == "n" else f"{k}={v:.2f}"
                                    for k, v in m.items()))


if __name__ == "__main__":
    main()
