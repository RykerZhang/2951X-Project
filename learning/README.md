# Learning lane: learned I' and T'

This folder learns the initiation condition I'_o and the termination condition T'_o of every
skill with decision trees, as described in section 2.3 of the main README. `TreeModel` is a
drop-in replacement for `HandFlawedModel`.

| File | Content |
|---|---|
| `data.py` | Runs the skills for real and collects the training data; generates random start states |
| `models.py` | `TrueModel`, `HandFlawedModel`, and the decision-tree `TreeModel` |
| `report.py` | Prints the learned rules and the held-out accuracy of every skill |

## How the data is made

1. Put the agent in a start state and really run the skill with its true termination, for at
   most `MAX_STEPS` actions.
2. The features of the start state plus "did the run reach the subgoal" is one I' example.
   The features of every visited state plus "is the subgoal reached here" is one T' example.
3. Repeat until every skill has `n` I' examples.

The labels come only from running the policy and from `skill.subgoal`. `skill.initiation` is
never called: copying the true I into the labels would be cheating.

Where the start states come from is the `source` knob:

- `chain`: the states in which the skill is invoked when the chains in `CHAINS` are executed,
  i.e. the states the agent meets in the experiments.
- `random`: a random position and inventory.

A start state in which the subgoal already holds gives no I' example: the run does nothing
and still "succeeds", which says nothing about whether the policy works. It still gives a T'
example.

## Usage

```bash
pip install scikit-learn
python -m pytest -q                                    # 33 passed (13 passed, 1 skipped without sklearn)
python -m experiments.run --model tree --noise 0.5     # experiments with the learned rules
python -m learning.report --noise 0.5                  # learned rules and accuracy per skill
```

```python
from learning.models import TreeModel
model = TreeModel(seed=0, n=200, layout_noise=0.5)   # trains here; build one per seed, reuse it for all runs
```

| Knob | Meaning | Default |
|---|---|---|
| `n` | Real runs per skill, i.e. the number of I' examples | 200 |
| `source` | Where the start states come from: `chain` or `random` | `chain` |
| `learn` | `both`; or only `I` / only `T`, with the other one using the true rule | `both` |
| `p_start` | Start only if the tree's estimated success probability is ≥ `p_start`. Lower is more optimistic | 0.5 |
| `p_done` | Stop as soon as the tree's estimated probability that the subgoal is reached is ≥ `p_done`. Lower stops earlier | 0.5 |
| `layout_noise` etc. | Randomness of the training worlds, passed to `Env` as is. `experiments.run` uses the same randomness as the experiment | 0 |

- On the command line the knobs are `--n`, `--source`, `--learn`, `--p-start`, `--p-done`; their
  values appear in the log file name.
- The same seed and knobs give exactly the same trees. The worlds used for training, held-out
  evaluation and experiments never share a seed.
- `python -m learning.report` scores the rules on both start-state distributions. **Read both**:
  rules that are accurate on random states can be wrong on the states the chain hands over.
- After adding a new item to the world, add it to `INVENTORY_MAX` in `data.py`; otherwise a test
  fails and says so.
- With `source="chain"`, a skill that appears in no chain (currently `go_forest`) has no training
  data, and querying it raises an error.

## Sanity-check results

Furnace chain, 10 seeds × 50 runs. Like section 6.5 of the main README, these only check that
the code behaves sensibly; they are not experimental conclusions.

| Condition | Result |
|---|---|
| `chain`, n=200, no randomness | ok 100% |
| `chain`, n=200, noise 0.5 | ok 25%, L1 12%, L2 27%, L4 detected 37% |
| same, `--p-start 0.3` | ok 31%, L2 32%, L4 detected 37% (same as the true rules) |
| same, `--p-start 0.7` | ok 1%, L1 68%, L4 detected 32% |
| `chain`, n=5, noise 0.5 | ok 13%, L1 24%, L2 20%, L3 early 10%, L4 detected 18%, L4 undetected 6%, other 8% |
| `random`, n=200, no randomness | ok 30%, L1 70% |
| `random`, n=200, noise 0.5 | ok 10%, L1 45%, L2 12%, L4 detected 33% |

## Open questions for the group

1. **Default start states: `chain` or `random`.** With `random`, the I' of `mine_stone` is learned
   as "cannot start": whether it succeeds depends on the agent's row, f(s) has no row, and most
   random starts fail. Inside the chain the agent is often on the right row, so the chain keeps
   getting stuck with L1, and more data does not help. `chain` avoids this, but then the
   "learned rules + no randomness" cell has no failures at all.
2. **L1 and L2 at `mine_stone` are two sides of the same thing.** Its success rate inside the
   chain is close to one half. With `p_start` below it the skill always starts (failures count as
   L2); above it the skill never starts (L1). Around 0.5 the result varies a lot between seeds
   (L1 in 0 to 38 of 50 runs), so `p_start` must be fixed before looking at results, and 10 seeds
   may not be enough.
3. **T' can learn rules that are equivalent inside the chain but mean the wrong thing.** For
   example, seed 0 learns "done when planks ≤ 1" for `make_pickaxe`. Across the 10 seeds, 5 trees
   look at the pickaxe, 3 at the planks and 2 at the durability. All are correct in the current
   chains, but in a different chain some would stop early: a natural source of L3.
