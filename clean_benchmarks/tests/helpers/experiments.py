"""Loading an experiment file of experiments/."""

import os

from bench import experiment as E
from bench import runner


CB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(CB, "tests", "fixtures", "experiments")


def _exp(name="dense_coyo", sets=()):
    path = os.path.join(CB, E.path_of(name))
    raw = runner._set(E.load_raw(path), list(sets))
    return E.finish(raw, path)
