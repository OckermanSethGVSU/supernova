"""A fixture experiment planned into a run dir."""

import os

from bench import experiment as E


CB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(CB, "tests", "fixtures", "experiments")


def _load(name, sets, sweep=False):
    from bench import runner
    # E.EXPERIMENTS_DIR: where experiment lookups point (tests/conftest.py sets the test copies)
    path = os.path.join(E.EXPERIMENTS_DIR, "sweeps", name + ".yaml") if sweep else os.path.join(CB, E.path_of(name))
    raw = runner._set(E.load_raw(path), sets)
    if sweep:
        raw, variants = E.sweep_variants(raw, path)
        return E.finish(raw, path), variants
    return E.finish(raw, path)
