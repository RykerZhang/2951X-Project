"""The world (owned by the world owner).

Only defines what the world looks like and what can be done in it: map, state, the three
primitive actions, randomness, and the features f(s). No skills appear here. Map size, items
and recipes are placeholders and can change once the environment spec is fixed; the signatures
of State / step / features are a shared contract -- announce any change in the group chat first.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

# ---------- Map: a 10 x 5 grid with three regions (placeholder) ----------
WIDTH, HEIGHT = 10, 5
REGIONS = {"forest": range(0, 4), "base": range(4, 6), "mine": range(6, 10)}
REGION_ID = {"forest": 0, "base": 1, "mine": 2}

# ---------- Tech tree (placeholder) ----------
RECIPES = {
    "plank":   {"needs": {"wood": 1},  "station": None},
    "pickaxe": {"needs": {"plank": 2}, "station": "workbench"},
    "furnace": {"needs": {"stone": 3}, "station": "workbench"},
}

# Default layout with no randomness: everything is on row 2; every chain should succeed here
DEFAULT_TREES = [(0, 2), (2, 2), (1, 0)]
DEFAULT_WORKBENCH = (4, 2)
DEFAULT_STONE = (9, 2)


@dataclass
class State:
    pos: tuple[int, int]
    inventory: dict[str, int]
    durability: dict[str, int]
    stations: dict[str, tuple[int, int]]
    resources: dict[tuple[int, int], str]


@dataclass(frozen=True)
class Move:
    dx: int
    dy: int


@dataclass(frozen=True)
class Collect:
    pass


@dataclass(frozen=True)
class Craft:
    item: str


def region(pos) -> str:
    return next(name for name, xs in REGIONS.items() if pos[0] in xs)


def adjacent(pos, target) -> bool:
    """On the target or right next to it (Manhattan distance <= 1)."""
    return abs(pos[0] - target[0]) + abs(pos[1] - target[1]) <= 1


class Env:
    """Randomness knobs:
    layout_noise       probability that each object (tree, workbench, stone) is moved to a
                       random cell in its region
    collect_fail       probability that a single collect action fails
    pickaxe_durability durability of the pickaxe; each mining action costs 1
    With the knobs at 0 / 0 / large enough, the world is fully deterministic. All randomness
    goes through self.rng so runs are reproducible.
    """

    def __init__(self, seed=0, layout_noise=0.0, collect_fail=0.0, pickaxe_durability=10):
        self.rng = random.Random(seed)
        self.layout_noise = layout_noise
        self.collect_fail = collect_fail
        self.pickaxe_durability = pickaxe_durability

    def _place(self, default, region_name):
        if self.rng.random() < self.layout_noise:
            xs = REGIONS[region_name]
            return (self.rng.choice(list(xs)), self.rng.randrange(HEIGHT))
        return default

    def reset(self, pos=(1, 2), inventory=None) -> State:
        resources = {}
        for t in DEFAULT_TREES:
            resources[self._place(t, "forest")] = "tree"
        resources[self._place(DEFAULT_STONE, "mine")] = "stone"
        workbench = self._place(DEFAULT_WORKBENCH, "base")
        inv = dict(inventory or {})
        dur = {"pickaxe": self.pickaxe_durability} if inv.get("pickaxe") else {}
        return State(pos, inv, dur, {"workbench": workbench}, resources)

    def step(self, s: State, a) -> tuple[State, bool]:
        """Apply one action and return (new state, success). Does not modify the input state."""
        s = copy.deepcopy(s)
        if isinstance(a, Move):
            nx, ny = s.pos[0] + a.dx, s.pos[1] + a.dy
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                s.pos = (nx, ny)
                return s, True
            return s, False

        if isinstance(a, Collect):
            kind = s.resources.get(s.pos)
            if kind is None:
                return s, False
            if kind == "stone":
                if s.inventory.get("pickaxe", 0) < 1 or s.durability.get("pickaxe", 0) <= 0:
                    return s, False
                s.durability["pickaxe"] -= 1
            if self.rng.random() < self.collect_fail:
                return s, False
            item = {"tree": "wood", "stone": "stone"}[kind]
            s.inventory[item] = s.inventory.get(item, 0) + 1
            return s, True

        if isinstance(a, Craft):
            r = RECIPES[a.item]
            if r["station"] and not adjacent(s.pos, s.stations[r["station"]]):
                return s, False
            if any(s.inventory.get(k, 0) < v for k, v in r["needs"].items()):
                return s, False
            for k, v in r["needs"].items():
                s.inventory[k] -= v
            s.inventory[a.item] = s.inventory.get(a.item, 0) + 1
            if a.item == "pickaxe":
                s.durability["pickaxe"] = self.pickaxe_durability
            return s, True

        raise ValueError(f"unknown action {a}")


# ---------- Features f(s): all that initiation sets and learned models may see ----------
FEATURE_NAMES = ["wood", "plank", "stone", "pickaxe", "furnace",
                 "durability", "region", "adj_workbench"]


def features(s: State) -> dict:
    inv = s.inventory
    return {
        "wood": inv.get("wood", 0),
        "plank": inv.get("plank", 0),
        "stone": inv.get("stone", 0),
        "pickaxe": inv.get("pickaxe", 0),
        "furnace": inv.get("furnace", 0),
        "durability": s.durability.get("pickaxe", 0),
        "region": REGION_ID[region(s.pos)],
        "adj_workbench": int(adjacent(s.pos, s.stations["workbench"])),
    }
