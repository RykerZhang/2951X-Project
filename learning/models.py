"""Learned I'_o and T'_o (owned by the learning lane).

The controller talks to a model only through these two functions:
    can_start(skill, s) -> bool   I'_o
    is_done(skill, s)   -> bool   T'_o
Learned models may only use features(s) internally, never the full state.
"""
from env.world import features, FEATURE_NAMES
from learning.data import MAX_STEPS, collect, vec


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
    """I'_o and T'_o learned by decision trees from real runs of the skills (learning/data.py).
    Two sklearn DecisionTreeClassifiers per skill; both only ever see vec(s) = f(s).

    Knobs
        n           number of real runs (= I' examples) per skill
        source      where the start states of those runs come from: "chain" or "random"
        learn       "both", or only "I" / only "T" (the other one then uses the true rule)
        p_start     can_start is True iff the I' tree's estimate of P(run succeeds) >= p_start;
                    lower = more optimistic (0 never refuses, 1 starts only where every training
                    run in that leaf succeeded)
        p_done      is_done is True iff the T' tree's estimate of P(subgoal reached) >= p_done;
                    lower = stops earlier
        min_leaf    minimum number of training examples per leaf (sklearn's min_samples_leaf).
                    1 = fully grown tree. Larger values give shorter rules and smoother
                    probabilities, but a tree cannot split at all with fewer than 2 * min_leaf
                    examples, so do not raise it when n is small
        drop_vacuous  leave out of the I' data the runs that start with the subgoal already
                    reached (see learning/data.py)
        env_kwargs  randomness of the training worlds, e.g. layout_noise=0.5; pass the same
                    values as the Env of the experiment

    Same seed and knobs -> same trees. Training happens once, in __init__: build ONE model per
    seed and reuse it for every run of that seed. Per-skill accuracy and the printed rules:
    python -m learning.report
    """

    def __init__(self, seed=0, n=200, source="chain", learn="both", p_start=0.5, p_done=0.5,
                 min_leaf=1, drop_vacuous=True, max_steps=MAX_STEPS, **env_kwargs):
        # sklearn is imported here, not at the top, so the true / flawed models run without it
        from sklearn.tree import DecisionTreeClassifier
        if learn not in ("both", "I", "T"):
            raise ValueError(f"unknown learn {learn!r}; use 'both', 'I' or 'T'")
        self.source, self.learn, self.p_start, self.p_done = source, learn, p_start, p_done
        self.data = collect(source, n, seed, drop_vacuous=drop_vacuous, max_steps=max_steps,
                            **env_kwargs)
        self.trees = {"I": {}, "T": {}}
        for name, d in self.data.items():
            for kind in ("I", "T"):
                if d[kind]:
                    X, y = zip(*d[kind])
                    self.trees[kind][name] = DecisionTreeClassifier(
                        min_samples_leaf=min_leaf, random_state=seed).fit(X, y)
        self._cache = {}

    def prob(self, kind, name, x) -> float:
        """The tree's estimate for skill `name` at feature vector x:
        kind "I" -> P(a run started here reaches the subgoal), kind "T" -> P(subgoal reached)."""
        tree = self.trees[kind].get(name)
        if tree is None:
            raise RuntimeError(f"TreeModel has no {kind}' training data for skill {name!r} "
                               f"(source={self.source!r}); it was never run during training")
        key = (kind, name, x)
        if key not in self._cache:
            classes = list(tree.classes_)     # may be a single class if all labels were equal
            self._cache[key] = float(tree.predict_proba([x])[0][classes.index(1)]) \
                if 1 in classes else 0.0
        return self._cache[key]

    def can_start(self, skill, s) -> bool:
        if self.learn == "T":
            return super().can_start(skill, s)
        return self.prob("I", skill.name, vec(s)) >= self.p_start

    def is_done(self, skill, s) -> bool:
        if self.learn == "I":
            return super().is_done(skill, s)
        return self.prob("T", skill.name, vec(s)) >= self.p_done

    def rules(self, kind, name) -> str:
        """The learned tree for skill `name` as readable text (kind "I" or "T")."""
        from sklearn.tree import export_text
        tree = self.trees[kind].get(name)
        if tree is None:
            return "(no training data)\n"
        if len(tree.classes_) == 1:
            return f"|--- always {int(tree.classes_[0])}\n"
        return export_text(tree, feature_names=FEATURE_NAMES, show_weights=True)
