"""Tests for the learned I' / T' (learning lane): learning/data.py, TreeModel, learning/report.py.
Skipped when scikit-learn is not installed."""
import json
import random

import pytest

pytest.importorskip("sklearn")

from env.world import Env, features, FEATURE_NAMES
from skills.library import SKILLS, CHAINS
from experiments import run as experiments_run
from experiments.run import in_D, run_chain
from diagnostics.label import label
from learning import data as learning_data
from learning.data import collect, random_state, true_run, vec
from learning.models import TreeModel
from learning.report import evaluate

IN_CHAINS = sorted({name for chain in CHAINS.values() for name in chain})
COL = FEATURE_NAMES.index


def random_states(k, noise=0.5, seed=0):
    rng = random.Random(seed)
    return [random_state(Env(seed=i, layout_noise=noise), rng) for i in range(k)]


# ---------- the data really comes from running the skills ----------

def test_max_steps_matches_the_experiment_loop():
    assert learning_data.MAX_STEPS == experiments_run.MAX_STEPS


def test_true_run_agrees_with_in_D():
    """The learner's "did the run succeed" is the same ground truth the labels use (D_o)."""
    rng = random.Random(0)
    for name, skill in SKILLS.items():
        for i in range(40):
            env = Env(seed=i, layout_noise=0.5)
            s = random_state(env, rng)
            expected = in_D(env, skill, s)          # runs on a copy; env is untouched
            assert true_run(env, skill, s)[1] == expected, (name, i)


def test_labels_never_use_the_true_initiation(monkeypatch):
    """Copying the true I into the training labels would be cheating (README 2.3)."""
    def boom(self, f):
        raise AssertionError("training / learned rules must not call skill.initiation")
    for skill in SKILLS.values():
        monkeypatch.setattr(type(skill), "initiation", boom)
    s = Env(seed=0).reset()
    for source in ("chain", "random"):
        model = TreeModel(seed=0, n=10, source=source, layout_noise=0.5)
        for name in IN_CHAINS:
            model.can_start(SKILLS[name], s)
            model.is_done(SKILLS[name], s)


def test_learned_model_sees_only_features():
    """Two states with the same f(s) get the same answer, even when only one of them is in D.
    This gap between f(s) and the full state is where L2 comes from."""
    env, skill = Env(seed=0), SKILLS["mine_stone"]
    on_row = env.reset(pos=(7, 2), inventory={"pickaxe": 1})     # stone is at (9, 2)
    off_row = env.reset(pos=(7, 0), inventory={"pickaxe": 1})
    assert vec(on_row) == vec(off_row)
    assert in_D(env, skill, on_row) and not in_D(env, skill, off_row)
    model = TreeModel(seed=0, n=30, source="random")
    assert model.can_start(skill, on_row) == model.can_start(skill, off_row)
    assert model.is_done(skill, on_row) == model.is_done(skill, off_row)


def test_vacuous_starts_give_no_I_example():
    """A start state where the subgoal already holds says nothing about the policy."""
    data = collect("random", 50, seed=0)
    assert all(x[COL("pickaxe")] == 0 for x, _ in data["make_pickaxe"]["I"])
    assert all(x[COL("region")] != 1 for x, _ in data["go_base"]["I"])
    assert any(y for _, y in data["make_pickaxe"]["T"])          # T' still sees them
    kept = collect("random", 50, seed=0, drop_vacuous=False)
    assert any(x[COL("pickaxe")] >= 1 for x, _ in kept["make_pickaxe"]["I"])


def test_random_states_vary_every_feature():
    seen = {k: set() for k in FEATURE_NAMES}
    for s in random_states(300, noise=1.0):
        for k, v in features(s).items():
            seen[k].add(v)
    constant = [k for k, values in seen.items() if len(values) < 2]
    assert not constant, f"random_state never varies {constant}: add the item to INVENTORY_MAX"


# ---------- amount of data, reproducibility, separation of worlds ----------

def test_n_is_the_number_of_I_examples_per_skill():
    for source, expected in (("random", list(SKILLS)), ("chain", IN_CHAINS)):
        data = collect(source, 15, seed=0, layout_noise=0.5)
        assert {name: len(d["I"]) for name, d in data.items()} == \
            {name: (15 if name in expected else 0) for name in SKILLS}


def test_skill_that_is_never_reached_warns_and_cannot_be_queried():
    chains = {"broken": ["make_pickaxe", "go_base"]}             # no planks: the chain always
    with pytest.warns(UserWarning, match="go_base"):             # breaks before go_base
        data = collect("chain", 3, seed=0, chains=chains)
    assert len(data["make_pickaxe"]["I"]) == 3 and not data["go_base"]["I"]

    model = TreeModel(seed=0, n=5, source="chain")               # go_forest is in no chain
    with pytest.raises(RuntimeError, match="go_forest"):
        model.can_start(SKILLS["go_forest"], Env(seed=0).reset())


def test_same_seed_same_data_and_same_trees():
    kw = dict(n=20, source="chain", layout_noise=0.5)
    assert collect(seed=3, **kw) == collect(seed=3, **kw)
    assert collect(seed=3, **kw) != collect(seed=4, **kw)
    a, b = TreeModel(seed=3, **kw), TreeModel(seed=3, **kw)
    for name in IN_CHAINS:
        for kind in ("I", "T"):
            assert a.rules(kind, name) == b.rules(kind, name)


def test_heldout_data_is_not_the_training_data():
    """Even with no randomness in the world (all layouts equal), the two splits must differ."""
    for source in ("chain", "random"):
        train = collect(source, 20, seed=0, layout_noise=0.5)
        assert collect(source, 20, seed=0, split="heldout", layout_noise=0.5) != train
    assert collect("random", 20, seed=0, split="heldout") != collect("random", 20, seed=0)


def test_training_heldout_and_experiment_worlds_never_share_a_seed(monkeypatch):
    seeds = {"train": set(), "heldout": set()}
    for split in seeds:
        class Spy(Env):
            def __init__(self, seed=0, **kw):
                seeds[split].add(seed)
                super().__init__(seed=seed, **kw)
        monkeypatch.setattr(learning_data, "Env", Spy)
        collect("chain", 10, seed=0, split=split, layout_noise=0.5)
    assert seeds["train"] and seeds["heldout"]
    assert not seeds["train"] & seeds["heldout"]
    assert min(seeds["train"] | seeds["heldout"]) >= 10**9       # experiments: seed * 100_000 + run


# ---------- knobs ----------

def test_thresholds_are_monotone_and_zero_never_refuses():
    kw = dict(seed=0, n=100, source="chain", layout_noise=0.5)
    low, mid, high = (TreeModel(p_start=p, p_done=p, **kw) for p in (0.0, 0.5, 1.0))
    for s in random_states(60):
        for name in IN_CHAINS:
            skill = SKILLS[name]
            assert low.can_start(skill, s)
            assert high.can_start(skill, s) <= mid.can_start(skill, s) <= low.can_start(skill, s)
            assert high.is_done(skill, s) <= mid.is_done(skill, s) <= low.is_done(skill, s)


def test_learn_switch_keeps_the_other_rule_true():
    kw = dict(seed=0, n=20, source="chain", layout_noise=0.5)
    only_I, only_T = TreeModel(learn="I", **kw), TreeModel(learn="T", **kw)
    for s in random_states(60):
        for name in IN_CHAINS:
            skill = SKILLS[name]
            assert only_I.is_done(skill, s) == skill.subgoal(s)
            assert only_T.can_start(skill, s) == skill.initiation(features(s))


def test_unknown_knob_values_are_rejected():
    with pytest.raises(ValueError):
        TreeModel(learn="neither")
    with pytest.raises(ValueError):
        TreeModel(source="somewhere")


# ---------- the trees learn what they should ----------

def test_single_class_data_gives_a_constant_rule():
    model = TreeModel(seed=0, n=10, source="chain")              # no noise: every run succeeds
    assert model.rules("I", "get_wood") == "|--- always 1\n"
    assert model.can_start(SKILLS["get_wood"], Env(seed=0).reset())


def test_rules_that_depend_only_on_features_are_recovered():
    """Random starts, deterministic world: crafting succeeds iff a condition on f(s) holds, and
    every subgoal is a threshold on f(s), so enough data must give (almost) perfect trees."""
    model = TreeModel(seed=0, n=300, source="random")
    scores = evaluate(model, collect("random", 300, seed=0, split="heldout"))
    for name in ("make_planks", "make_pickaxe", "make_furnace"):
        assert scores[name]["I"]["acc"] >= 0.98, (name, scores[name]["I"])
    for name in SKILLS:
        assert scores[name]["T"]["acc"] == 1.0, (name, scores[name]["T"])


def test_more_data_gives_better_trees():
    crafting = {name: SKILLS[name] for name in ("make_planks", "make_pickaxe", "make_furnace")}

    def mean_acc(n):
        accs = []
        for seed in range(3):
            scores = evaluate(TreeModel(seed=seed, n=n, source="random"),
                              collect("random", 200, seed=seed, split="heldout", skills=crafting))
            accs += [scores[name]["I"]["acc"] for name in crafting]
        return sum(accs) / len(accs)
    assert mean_acc(100) > mean_acc(5)


def test_report_fractions_add_up():
    model = TreeModel(seed=0, n=50, source="chain", layout_noise=0.5)
    scores = evaluate(model, collect("random", 50, seed=0, split="heldout", layout_noise=0.5))
    for name in IN_CHAINS:
        i, t = scores[name]["I"], scores[name]["T"]
        assert i["acc"] + i["false_accept"] + i["false_reject"] == pytest.approx(1)
        assert t["acc"] + t["false_done"] + t["missed_done"] == pytest.approx(1)


# ---------- TreeModel in the execution loop ----------

def test_chains_succeed_with_trees_in_the_deterministic_world():
    """Validation condition for the learned rules: trained on chain states with no randomness,
    every chain still succeeds."""
    model = TreeModel(seed=0, n=20, source="chain")
    for chain in CHAINS.values():
        env = Env(seed=0)
        assert label(run_chain(env, chain, env.reset(), model))[0] == "ok"


def test_tree_model_is_a_drop_in_for_the_execution_loop():
    model = TreeModel(seed=0, n=50, source="chain", layout_noise=0.5)
    labels = set()
    for i in range(40):
        env = Env(seed=i, layout_noise=0.5)
        records = run_chain(env, CHAINS["furnace"], env.reset(), model)
        json.dumps(records)                                      # log records stay JSON-serializable
        labels.add(label(records)[0])
    assert labels <= {"ok", "L1", "L2", "L3_early", "L3_late", "L4", "other"}
    assert len(labels) > 1
