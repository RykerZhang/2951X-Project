from env.world import Env
from skills.library import SKILLS, CHAINS
from learning.models import TrueModel
from experiments.run import run_chain, in_D
from diagnostics.label import label


def test_validation_condition():
    """Validation condition: with true I/T and no randomness, every chain succeeds
    (L1-L3 must be 0)."""
    for chain in CHAINS.values():
        env = Env(seed=0, layout_noise=0.0)
        records = run_chain(env, chain, env.reset(), TrueModel())
        assert label(records)[0] == "ok", records


def test_each_skill_from_D_reaches_subgoal():
    """Every skill, started from a state in D with no randomness, always reaches its subgoal."""
    starts = {
        "get_wood": dict(pos=(1, 2)),
        "make_planks": dict(pos=(1, 2), inventory={"wood": 2}),
        "go_base": dict(pos=(1, 2)),
        "make_pickaxe": dict(pos=(4, 2), inventory={"plank": 2}),
        "go_mine": dict(pos=(4, 2)),
        "mine_stone": dict(pos=(6, 2), inventory={"pickaxe": 1}),
        "make_furnace": dict(pos=(4, 2), inventory={"stone": 3}),
    }
    for name, kw in starts.items():
        env = Env(seed=0)
        assert in_D(env, SKILLS[name], env.reset(**kw)), name
