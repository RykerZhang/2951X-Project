"""Training data for the learned I'_o and T'_o (owned by the learning lane).

One sample unit is one REAL RUN of a skill: put the agent in a start state, execute the skill's
policy until its true subgoal holds or MAX_STEPS actions are used up, and record
    I' example    (f(start state), did the run reach the subgoal?)
    T' examples   (f(s), is the subgoal reached in s?)  for every state s visited in the run
Labels come only from running the policy and from skill.subgoal (judged by the simulator).
skill.initiation is never called here: copying the true I into the labels would be cheating.

Where the start states come from is the `source` knob:
    "chain"   the states in which the skill is invoked when the chains in CHAINS are executed
              (with true termination), i.e. the states the agent meets in the experiments
    "random"  a uniformly random position and inventory (the recipe in README 2.3)

A start state in which the subgoal already holds is "vacuous": the run does nothing and still
"succeeds". Such a run says nothing about whether the policy works, so by default it gives no
I' example (drop_vacuous=True). It still gives a T' example.

This module uses only the shared contracts of env/ and skills/ and never imports experiments/.
"""
import random
import warnings

from env.world import Env, features, FEATURE_NAMES, WIDTH, HEIGHT
from skills.library import SKILLS, CHAINS

MAX_STEPS = 20   # must equal experiments.run.MAX_STEPS (tests/test_learning.py checks this)

# Largest count of each item that a random start state may hold. World owner: when you add an
# item to the world, add it here too (test_random_states_vary_every_feature fails until you do).
INVENTORY_MAX = {"wood": 4, "plank": 4, "stone": 4, "pickaxe": 1, "furnace": 1,
                 "iron_ore": 4, "iron": 4}

# Worlds used for training and for held-out evaluation never share a seed with each other, nor
# with the experiment worlds (experiments/run.py uses seeds far below 10**9).
ENV_SEED_BASE = {"train": 10**9, "heldout": 2 * 10**9}

# Give up after TRIES * n attempts if a skill still has fewer than n I' examples.
TRIES = 20


def vec(s) -> tuple:
    """The feature vector f(s), in FEATURE_NAMES order. All a learned model may look at."""
    f = features(s)
    return tuple(f[k] for k in FEATURE_NAMES)


def random_state(env, rng):
    """A uniformly random start state: any cell, any inventory within INVENTORY_MAX, any
    remaining tool durability. The layout comes from env.reset(), so it follows env's noise."""
    inv = {item: rng.randint(0, top) for item, top in INVENTORY_MAX.items()}
    s = env.reset(pos=(rng.randrange(WIDTH), rng.randrange(HEIGHT)), inventory=inv)
    for tool, full in s.durability.items():
        s.durability[tool] = rng.randint(0, full)
    return s


def true_run(env, skill, s, max_steps=MAX_STEPS):
    """Really execute `skill` from `s` with its TRUE termination. Same loop as
    experiments.run.in_D, but on the real env and keeping the visited states.
    Returns (visited states, reached the subgoal?)."""
    visited = [s]
    for t in range(max_steps):
        if skill.subgoal(s):
            return visited, True
        s, _ = env.step(s, skill.policy(s, t))
        visited.append(s)
    return visited, skill.subgoal(s)


def collect(source, n, seed, split="train", skills=SKILLS, chains=CHAINS,
            drop_vacuous=True, max_steps=MAX_STEPS, **env_kwargs) -> dict:
    """Collect n real runs per skill.

    source      "chain" or "random" (see the module docstring)
    n           number of I' examples per skill
    seed        same arguments + same seed -> same data
    split       "train" or "heldout"; the two never share a world
    env_kwargs  randomness of the worlds, passed to Env (e.g. layout_noise=0.5)

    Returns {skill name: {"I": [(x, y), ...], "T": [(x, y), ...]}} with x = vec(state), y in {0, 1}.
    With source="chain", a skill that appears in no chain gets no data. A skill that could not
    reach n examples within TRIES * n attempts keeps what it has, and a warning names it.
    """
    if source not in ("chain", "random"):
        raise ValueError(f"unknown source {source!r}; use 'chain' or 'random'")
    rng = random.Random(f"{split}/{seed}")          # train and heldout draw different states too
    data = {name: {"I": [], "T": []} for name in skills}

    def new_env():
        return Env(seed=ENV_SEED_BASE[split] + rng.randrange(10**9), **env_kwargs)

    def full(name):
        return len(data[name]["I"]) >= n

    def run(env, skill, s):
        """Run `skill` from `s` for real; store the examples unless the skill is already full.
        Returns (end state, reached the subgoal?)."""
        vacuous = skill.subgoal(s)
        visited, ok = true_run(env, skill, s, max_steps)
        if not full(skill.name):
            if not (vacuous and drop_vacuous):
                data[skill.name]["I"].append((vec(s), int(ok)))
            data[skill.name]["T"].extend((vec(v), int(skill.subgoal(v))) for v in visited)
        return visited[-1], ok

    if source == "random":
        wanted = list(skills)
        for name in wanted:
            for _ in range(TRIES * n):
                if full(name):
                    break
                env = new_env()
                run(env, skills[name], random_state(env, rng))
    else:
        wanted = sorted({name for chain in chains.values() for name in chain})
        for _ in range(TRIES * n):
            todo = [chain for chain in chains.values() if not all(full(name) for name in chain)]
            if not todo:
                break
            for chain in todo:                      # one episode of every chain that still needs data
                env = new_env()
                s = env.reset()
                for name in chain:
                    s, ok = run(env, skills[name], s)
                    if not ok:                      # the chain breaks here; later skills are not reached
                        break

    short = {name: len(data[name]["I"]) for name in wanted if not full(name)}
    if short:
        warnings.warn(f"collect(source={source!r}, n={n}): fewer than n I' examples for {short}; "
                      f"these skills are rarely reached or their subgoal holds almost everywhere")
    return data
