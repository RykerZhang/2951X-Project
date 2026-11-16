from env.world import Env, Craft, Move, Collect


def test_craft_pickaxe_needs_workbench():
    env = Env(seed=0)
    s = env.reset(pos=(1, 2), inventory={"plank": 2})
    _, ok = env.step(s, Craft("pickaxe"))
    assert ok is False


def test_craft_pickaxe_next_to_workbench():
    env = Env(seed=0)
    s = env.reset(pos=(4, 2), inventory={"plank": 2})
    s2, ok = env.step(s, Craft("pickaxe"))
    assert ok and s2.inventory["pickaxe"] == 1


def test_cannot_walk_off_map():
    env = Env(seed=0)
    s = env.reset(pos=(0, 2))
    _, ok = env.step(s, Move(-1, 0))
    assert ok is False


def test_same_seed_same_world():
    a, b = Env(seed=7, layout_noise=1.0).reset(), Env(seed=7, layout_noise=1.0).reset()
    assert a == b


def test_mining_uses_durability():
    env = Env(seed=0, pickaxe_durability=1)
    s = env.reset(pos=(9, 2), inventory={"pickaxe": 1})
    s, ok1 = env.step(s, Collect())
    _, ok2 = env.step(s, Collect())
    assert ok1 and not ok2
