"""Skill library (owned by the skills owner).

Each skill defines only three functions and only guarantees its own subgoal; it does not know
which skill comes before or after it:
    subgoal(s)    T_o: is the subgoal reached? (full state, judged by the simulator)
    initiation(f) I_o: can the skill start? (may only look at the features f(s))
    policy(s, t)  what to do at step t (a fixed script)
The execution loop lives in experiments/run.py; skills never run themselves.

Shared design conventions (follow them; never change a skill to create or avoid a break type):
    - Navigation skills go_xxx stop as soon as they enter the target region; they do not walk
      to any particular cell inside it
    - Crafting skills craft in place and never move
    - Collecting skills look for the resource in the current region
"""
from env.world import Move, Collect, Craft, State, region


class Skill:
    name = ""

    def subgoal(self, s: State) -> bool:
        raise NotImplementedError

    def initiation(self, f: dict) -> bool:
        raise NotImplementedError

    def policy(self, s: State, t: int):
        raise NotImplementedError


# ---------- Navigation ----------
class GoTo(Skill):
    def __init__(self, target: str):
        self.target = target
        self.name = f"go_{target}"

    def subgoal(self, s):    return region(s.pos) == self.target
    def initiation(self, f): return True

    def policy(self, s, t):
        order = ["forest", "base", "mine"]
        here, there = order.index(region(s.pos)), order.index(self.target)
        return Move(1 if there > here else -1, 0)


# ---------- Collecting ----------
class GetWood(Skill):
    name = "get_wood"
    def subgoal(self, s):    return s.inventory.get("wood", 0) >= 2
    def initiation(self, f): return f["region"] == 0

    def policy(self, s, t):
        if s.resources.get(s.pos) == "tree":
            return Collect()
        trees = [p for p, k in s.resources.items() if k == "tree"]
        tx, ty = min(trees, key=lambda p: abs(p[0] - s.pos[0]) + abs(p[1] - s.pos[1]))
        if tx != s.pos[0]:
            return Move(1 if tx > s.pos[0] else -1, 0)
        return Move(0, 1 if ty > s.pos[1] else -1)


class MineStone(Skill):
    name = "mine_stone"
    def subgoal(self, s):    return s.inventory.get("stone", 0) >= 3
    def initiation(self, f): return f["pickaxe"] >= 1 and f["region"] == 2

    def policy(self, s, t):
        # Design assumption: the stone is on the row where the agent enters the mine, so walking
        # east will reach it. I does not encode this assumption -> when the stone is on another
        # row, the state satisfies I but is not in D (L2)
        if s.resources.get(s.pos) == "stone":
            return Collect()
        return Move(1, 0)


# ---------- Crafting ----------
class MakePlanks(Skill):
    name = "make_planks"
    def subgoal(self, s):    return s.inventory.get("plank", 0) >= 2
    def initiation(self, f): return f["wood"] >= 2
    def policy(self, s, t):  return Craft("plank")


class MakePickaxe(Skill):
    name = "make_pickaxe"
    def subgoal(self, s):    return s.inventory.get("pickaxe", 0) >= 1
    def initiation(self, f): return f["plank"] >= 2 and f["adj_workbench"] == 1
    def policy(self, s, t):  return Craft("pickaxe")


class MakeFurnace(Skill):
    name = "make_furnace"
    def subgoal(self, s):    return s.inventory.get("furnace", 0) >= 1
    def initiation(self, f): return f["stone"] >= 3 and f["adj_workbench"] == 1
    def policy(self, s, t):  return Craft("furnace")


# TODO skills owner: add the remaining skills from the environment spec, e.g. mine_iron and
#      smelt_iron, and add a unit test for every new skill in tests/test_skills.py.

SKILLS = {sk.name: sk for sk in [
    GoTo("forest"), GoTo("base"), GoTo("mine"),
    GetWood(), MineStone(), MakePlanks(), MakePickaxe(), MakeFurnace(),
]}

# Chains used in the experiments (fixed order; the controller just calls them in sequence)
CHAINS = {
    "pickaxe": ["get_wood", "make_planks", "go_base", "make_pickaxe"],
    "stone":   ["get_wood", "make_planks", "go_base", "make_pickaxe", "go_mine", "mine_stone"],
    "furnace": ["get_wood", "make_planks", "go_base", "make_pickaxe", "go_mine", "mine_stone",
                "go_base", "make_furnace"],
}
