"""Learned I'_o and T'_o (owned by the learning lane).

The controller talks to a model only through these two functions:
    can_start(skill, s) -> bool   I'_o
    is_done(skill, s)   -> bool   T'_o
Learned models may only use features(s) internally, never the full state.
"""
from env.world import features


import random


class TrueModel:
    """Uses the true I_o and T_o. This is the "true rules" condition in the experiments,
    and is also used for the validation condition."""

    def __init__(self, seed=0):
        pass

    def can_start(self, skill, s) -> bool:
        return skill.initiation(features(s))

    def is_done(self, skill, s) -> bool:
        return skill.subgoal(s)


class HandFlawedModel(TrueModel):
    """A hand-written "wrongly learned" model, only for demonstrating what L1, L3 and
    undetected L4 look like. Replace it with TreeModel once the learning lane has decision trees.

    A really learned model would not make the same mistake every time, so each run randomly
    includes some of the errors: each error appears with probability p."""

    def __init__(self, seed=0, p=0.3):
        rng = random.Random(seed)
        self.forget_workbench = rng.random() < p   # make_pickaxe ignores the workbench -> undetected L4
        self.too_strict = rng.random() < p         # make_furnace requires 4 stones -> L1
        self.stop_early = rng.random() < p         # make_planks stops after 1 plank -> L3 early

    def can_start(self, skill, s) -> bool:
        f = features(s)
        if skill.name == "make_pickaxe" and self.forget_workbench:
            return f["plank"] >= 2
        if skill.name == "make_furnace" and self.too_strict:
            return f["stone"] >= 4 and f["adj_workbench"] == 1
        return super().can_start(skill, s)

    def is_done(self, skill, s) -> bool:
        if skill.name == "make_planks" and self.stop_early:
            return features(s)["plank"] >= 1
        return super().is_done(skill, s)


class TreeModel(TrueModel):
    """TODO learning lane:
    1. Run every skill from random states; collect (features, did the run succeed) and
       (features, is the subgoal reached)
    2. Train two sklearn DecisionTreeClassifiers per skill
    3. can_start / is_done use the trees' predictions and may only look at features(s)
    4. Two knobs: training-set size n; optimistic / pessimistic bias (decision threshold or
       class_weight)
    5. Report per skill: acc / false_accept / false_reject / n, and print the rules with
       export_text
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("TODO: learning lane has not implemented TreeModel yet")
