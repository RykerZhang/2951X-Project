"""Tests for the world (world owner). Only primitive actions are used here, never skills."""
import pytest

from env.world import (Env, Craft, Move, Collect, features, FEATURE_NAMES, REGIONS, region,
                       DEFAULT_TREES, DEFAULT_STONE, DEFAULT_IRON_ORE, DEFAULT_WORKBENCH,
                       DEFAULT_FURNACE)


def walk(env, s, dx, n):
    for _ in range(n):
        s, ok = env.step(s, Move(dx, 0))
        assert ok
    return s


def do(env, s, a, times=1):
    for _ in range(times):
        s, ok = env.step(s, a)
        assert ok, (a, s)
    return s


# ---------- existing behaviour ----------

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


def test_step_does_not_modify_its_input():
    env = Env(seed=0)
    s = env.reset(pos=(4, 2), inventory={"plank": 2})
    before = repr(s)
    env.step(s, Craft("pickaxe"))
    assert repr(s) == before


# ---------- iron ore ----------

def test_iron_ore_needs_a_pickaxe():
    env = Env(seed=0)
    s = env.reset(pos=DEFAULT_IRON_ORE)
    _, ok = env.step(s, Collect())
    assert ok is False


def test_mining_iron_ore_gives_ore_and_costs_durability():
    env = Env(seed=0, pickaxe_durability=2)
    s = env.reset(pos=DEFAULT_IRON_ORE, inventory={"pickaxe": 1})
    s, ok1 = env.step(s, Collect())
    assert ok1 and s.inventory["iron_ore"] == 1 and s.durability["pickaxe"] == 1
    s, ok2 = env.step(s, Collect())
    _, ok3 = env.step(s, Collect())
    assert ok2 and not ok3 and s.inventory["iron_ore"] == 2


def test_stone_and_iron_ore_share_the_durability():
    env = Env(seed=0, pickaxe_durability=2)
    s = env.reset(pos=DEFAULT_STONE, inventory={"pickaxe": 1})
    s = do(env, s, Collect(), 2)
    s = walk(env, s, -1, DEFAULT_STONE[0] - DEFAULT_IRON_ORE[0])
    _, ok = env.step(s, Collect())
    assert ok is False


def test_failed_collect_still_costs_durability():
    """Documented behaviour (see the env/world.py docstring)."""
    env = Env(seed=0, collect_fail=1.0)
    s = env.reset(pos=DEFAULT_STONE, inventory={"pickaxe": 1})
    s2, ok = env.step(s, Collect())
    assert not ok and s2.durability["pickaxe"] == s.durability["pickaxe"] - 1


# ---------- furnace station and smelting ----------

def test_crafting_a_furnace_places_the_station_where_the_agent_stands():
    env = Env(seed=0)
    s = env.reset(pos=(5, 2), inventory={"stone": 3})
    assert "furnace" not in s.stations
    s2, ok = env.step(s, Craft("furnace"))
    assert ok and s2.inventory["furnace"] == 1 and s2.inventory["stone"] == 0
    assert s2.stations["furnace"] == (5, 2)


def test_furnace_needs_the_workbench():
    env = Env(seed=0)
    s = env.reset(pos=(9, 2), inventory={"stone": 3})
    s2, ok = env.step(s, Craft("furnace"))
    assert not ok and "furnace" not in s2.stations


def test_smelting_fails_without_a_furnace():
    env = Env(seed=0)
    s = env.reset(pos=DEFAULT_FURNACE, inventory={"iron_ore": 1})
    _, ok = env.step(s, Craft("iron"))
    assert ok is False


def test_smelting_needs_to_be_next_to_the_furnace():
    env = Env(seed=0)
    far = env.reset(pos=(9, 0), inventory={"furnace": 1, "iron_ore": 1})
    _, ok = env.step(far, Craft("iron"))
    assert ok is False

    near = env.reset(pos=DEFAULT_FURNACE, inventory={"furnace": 1, "iron_ore": 1})
    s, ok = env.step(near, Craft("iron"))
    assert ok and s.inventory["iron"] == 1 and s.inventory["iron_ore"] == 0
    assert s.inventory["furnace"] == 1                          # the furnace is not used up


def test_smelting_needs_iron_ore():
    env = Env(seed=0)
    s = env.reset(pos=DEFAULT_FURNACE, inventory={"furnace": 1})
    _, ok = env.step(s, Craft("iron"))
    assert ok is False


def test_reset_places_a_furnace_only_if_the_inventory_has_one():
    env = Env(seed=0)
    assert "furnace" not in env.reset().stations
    assert env.reset(inventory={"furnace": 1}).stations["furnace"] == DEFAULT_FURNACE


def test_adj_furnace_feature():
    env = Env(seed=0)
    assert features(env.reset(pos=DEFAULT_FURNACE))["adj_furnace"] == 0       # no furnace yet
    s = env.reset(pos=DEFAULT_FURNACE, inventory={"furnace": 1})
    assert features(s)["adj_furnace"] == 1
    s = walk(env, s, 1, 3)
    assert features(s)["adj_furnace"] == 0


# ---------- randomness ----------

def test_collect_fail_knob():
    always, never = Env(seed=0, collect_fail=1.0), Env(seed=0, collect_fail=0.0)
    for env, expected in ((always, False), (never, True)):
        s = env.reset(pos=DEFAULT_TREES[0])
        for _ in range(20):
            s, ok = env.step(s, Collect())
            assert ok is expected


def test_no_noise_gives_the_default_layout():
    for seed in range(20):
        s = Env(seed=seed).reset(inventory={"furnace": 1})
        assert s.resources == {**{t: "tree" for t in DEFAULT_TREES},
                               DEFAULT_STONE: "stone", DEFAULT_IRON_ORE: "iron_ore"}
        assert s.stations == {"workbench": DEFAULT_WORKBENCH, "furnace": DEFAULT_FURNACE}


def test_noisy_layout_stays_in_region_and_never_overlaps():
    home = {"tree": "forest", "stone": "mine", "iron_ore": "mine",
            "workbench": "base", "furnace": "base"}
    for seed in range(300):
        s = Env(seed=seed, layout_noise=1.0).reset(inventory={"furnace": 1})
        kinds = sorted(s.resources.values())
        assert kinds == ["iron_ore", "stone", "tree", "tree", "tree"], (seed, s.resources)
        cells = list(s.resources) + list(s.stations.values())
        assert len(cells) == len(set(cells)), (seed, cells)
        for cell, kind in s.resources.items():
            assert region(cell) == home[kind]
        for kind, cell in s.stations.items():
            assert region(cell) == home[kind]


def test_noise_actually_moves_objects():
    moved = sum(Env(seed=seed, layout_noise=1.0).reset().resources.get(DEFAULT_IRON_ORE)
                != "iron_ore" for seed in range(50))
    assert moved > 25


# ---------- the whole tech tree ----------

def longest_chain_by_hand(env):
    """The 12-skill chain (wood -> planks -> pickaxe -> stone -> furnace -> iron), played with
    primitive actions on the default layout. Returns the final state."""
    s = env.reset()                                              # (1, 2)
    s = walk(env, s, -1, 1)                                      # tree at (0, 2)
    s = do(env, s, Collect(), 2)
    s = do(env, s, Craft("plank"), 2)
    s = walk(env, s, 1, 4)                                       # base, next to the workbench
    s = do(env, s, Craft("pickaxe"))
    s = walk(env, s, 1, 5)                                       # stone at (9, 2)
    s = do(env, s, Collect(), 3)
    s = walk(env, s, -1, 4)                                      # back in base at (5, 2)
    s = do(env, s, Craft("furnace"))
    s = walk(env, s, 1, 2)                                       # iron ore at (7, 2)
    s = do(env, s, Collect())
    s = walk(env, s, -1, 2)                                      # next to the furnace
    s = do(env, s, Craft("iron"))
    return s


def test_whole_tech_tree_is_reachable_without_noise():
    s = longest_chain_by_hand(Env(seed=0))
    assert s.inventory["iron"] == 1 and s.inventory["pickaxe"] == 1
    assert s.inventory["furnace"] == 1 and s.durability["pickaxe"] == 10 - 4


@pytest.mark.parametrize("noise", [0.0, 0.5])
def test_same_seed_same_episode(noise):
    """Determinism over a whole episode (collect rolls included), not just reset()."""
    def episode(seed):
        env = Env(seed=seed, layout_noise=noise, collect_fail=0.3)
        s, trace = env.reset(pos=(0, 2)), []
        for a in [Collect()] * 10 + [Move(1, 0)] * 3 + [Collect()] * 5:
            s, ok = env.step(s, a)
            trace.append((repr(s), ok))
        return trace
    assert episode(3) == episode(3)
    assert episode(3) != episode(4)


# ---------- features ----------

def test_features_follow_feature_names():
    f = features(Env(seed=0).reset())
    assert list(f) == FEATURE_NAMES


def test_old_features_keep_their_positions():
    """learning.data.vec() relies on the order; new features are only appended."""
    assert FEATURE_NAMES[:8] == ["wood", "plank", "stone", "pickaxe", "furnace",
                                 "durability", "region", "adj_workbench"]
