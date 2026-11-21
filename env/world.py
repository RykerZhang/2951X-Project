"""The world (owned by the world owner).

Only defines what the world looks like and what can be done in it: map, state, the three
primitive actions, randomness, and the features f(s). No skills appear here. The signatures
of State / step / features are a shared contract -- announce any change in the group chat first.

World spec (smelting option B, agreed at the group meeting):
    Items     wood, plank, stone, pickaxe, furnace, iron_ore, iron
    Stations  workbench  fixed object in the base, placed by reset()
              furnace    placed by crafting a furnace: it is built on the cell where the agent
                         stands (the agent must be next to the workbench to craft it).
                         inventory["furnace"] counts furnaces built; it is NOT a carried tool.
                         Building another furnace moves the one station to the new cell.
    Recipes   plank   <- 1 wood                    (anywhere)
              pickaxe <- 2 plank                   (next to the workbench)
              furnace <- 3 stone                   (next to the workbench; places the station)
              iron    <- 1 iron_ore                (next to the furnace)
    Collect   tree -> wood; stone -> stone; iron_ore -> iron_ore.
              Stone and iron ore need a pickaxe with durability > 0 and cost 1 durability.
    Resources never run out: one cell gives any number of items.

Behaviour that is deliberate (written down so nobody "fixes" it by accident):
    - Mining costs 1 durability even when the collect_fail roll then fails.
    - Crafting a pickaxe resets the durability to full; all pickaxes share one durability.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

# ---------- Map: a 10 x 5 grid with three regions ----------
WIDTH, HEIGHT = 10, 5
REGIONS = {"forest": range(0, 4), "base": range(4, 6), "mine": range(6, 10)}
REGION_ID = {"forest": 0, "base": 1, "mine": 2}

# ---------- Tech tree ----------
RECIPES = {
    "plank":   {"needs": {"wood": 1},     "station": None},
    "pickaxe": {"needs": {"plank": 2},    "station": "workbench"},
    "furnace": {"needs": {"stone": 3},    "station": "workbench"},
    "iron":    {"needs": {"iron_ore": 1}, "station": "furnace"},
}

# Resource cell -> item it gives, and which of them need a pickaxe
YIELDS = {"tree": "wood", "stone": "stone", "iron_ore": "iron_ore"}
NEEDS_PICKAXE = {"stone", "iron_ore"}

# Default layout with no randomness: everything is on row 2; every chain should succeed here
DEFAULT_TREES = [(0, 2), (2, 2), (1, 0)]
DEFAULT_WORKBENCH = (4, 2)
DEFAULT_STONE = (9, 2)
DEFAULT_IRON_ORE = (7, 2)
# Where reset() puts the furnace station when the start inventory already has a furnace
DEFAULT_FURNACE = (5, 2)


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
    layout_noise       probability that each object (tree, stone, iron ore, workbench, and a
                       furnace given in the start inventory) is moved to a random cell in its
                       region. Two objects never share a cell.
    collect_fail       probability that a single collect action fails
    pickaxe_durability durability of the pickaxe; each mining action (stone or iron ore) costs 1
    With the knobs at 0 / 0 / large enough, the world is fully deterministic. All randomness
    goes through self.rng so runs are reproducible.
    """

    def __init__(self, seed=0, layout_noise=0.0, collect_fail=0.0, pickaxe_durability=10):
        self.rng = random.Random(seed)
        self.layout_noise = layout_noise
        self.collect_fail = collect_fail
        self.pickaxe_durability = pickaxe_durability

    def _random_cell(self, region_name):
        return (self.rng.choice(list(REGIONS[region_name])), self.rng.randrange(HEIGHT))

    def _place(self, default, region_name, taken):
        """Default cell, or with probability layout_noise a random cell in the region.
        If the cell is already taken by another object, draw again until it is free.
        (With no noise nothing ever collides, so the default layout is unchanged.)"""
        cell = self._random_cell(region_name) if self.rng.random() < self.layout_noise \
            else default
        while cell in taken:
            cell = self._random_cell(region_name)
        taken.add(cell)
        return cell

    def reset(self, pos=(1, 2), inventory=None) -> State:
        # Order of placement is fixed (trees, stone, workbench, iron ore, furnace) so that for
        # a given seed the objects that existed before iron ore keep their old positions.
        taken = set()
        resources = {}
        for t in DEFAULT_TREES:
            resources[self._place(t, "forest", taken)] = "tree"
        resources[self._place(DEFAULT_STONE, "mine", taken)] = "stone"
        stations = {"workbench": self._place(DEFAULT_WORKBENCH, "base", taken)}
        resources[self._place(DEFAULT_IRON_ORE, "mine", taken)] = "iron_ore"
        inv = dict(inventory or {})
        if inv.get("furnace", 0) >= 1:
            stations["furnace"] = self._place(DEFAULT_FURNACE, "base", taken)
        dur = {"pickaxe": self.pickaxe_durability} if inv.get("pickaxe") else {}
        return State(pos, inv, dur, stations, resources)

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
            if kind in NEEDS_PICKAXE:
                if s.inventory.get("pickaxe", 0) < 1 or s.durability.get("pickaxe", 0) <= 0:
                    return s, False
                s.durability["pickaxe"] -= 1
            if self.rng.random() < self.collect_fail:
                return s, False
            item = YIELDS[kind]
            s.inventory[item] = s.inventory.get(item, 0) + 1
            return s, True

        if isinstance(a, Craft):
            r = RECIPES[a.item]
            station = r["station"]
            if station and (station not in s.stations
                            or not adjacent(s.pos, s.stations[station])):
                return s, False
            if any(s.inventory.get(k, 0) < v for k, v in r["needs"].items()):
                return s, False
            for k, v in r["needs"].items():
                s.inventory[k] -= v
            s.inventory[a.item] = s.inventory.get(a.item, 0) + 1
            if a.item == "pickaxe":
                s.durability["pickaxe"] = self.pickaxe_durability
            if a.item == "furnace":
                s.stations["furnace"] = s.pos
            return s, True

        raise ValueError(f"unknown action {a}")


# ---------- Features f(s): all that initiation sets and learned models may see ----------
# Append only: learning.data.vec() relies on this order.
FEATURE_NAMES = ["wood", "plank", "stone", "pickaxe", "furnace",
                 "durability", "region", "adj_workbench",
                 "iron_ore", "iron", "adj_furnace"]


def features(s: State) -> dict:
    inv = s.inventory
    furnace = s.stations.get("furnace")
    return {
        "wood": inv.get("wood", 0),
        "plank": inv.get("plank", 0),
        "stone": inv.get("stone", 0),
        "pickaxe": inv.get("pickaxe", 0),
        "furnace": inv.get("furnace", 0),
        "durability": s.durability.get("pickaxe", 0),
        "region": REGION_ID[region(s.pos)],
        "adj_workbench": int(adjacent(s.pos, s.stations["workbench"])),
        "iron_ore": inv.get("iron_ore", 0),
        "iron": inv.get("iron", 0),
        "adj_furnace": int(furnace is not None and adjacent(s.pos, furnace)),
    }
