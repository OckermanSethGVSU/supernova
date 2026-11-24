"""An experiment of experiments/ planned into a run dir, and the systems an experiment drops."""

import os

from bench import experiment as E


CB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(CB, "experiments")


# (Seth): the two native duplicate legs are dropped from filtered_fineweb_smoke's leg set and
# rotation, and so are the copy-only rungs (every GPU line copies pinned + overlapped by
# default, so each is its base line); everything else is the old order
DROPPED = {"filtered_fineweb_smoke": {"cuvs_bitmap_native_ovl", "cuvs_union_native_d2h", "cuvs_bitmap_ovl",
                                "cuvs_union_d2h", "cuvs_gather_d2h", "faiss_gather_d2h"}}


def _load(name, sets, sweep=False):
    from bench import runner
    path = os.path.join(EXP, "sweeps", name + ".yaml") if sweep else os.path.join(CB, E.path_of(name))
    raw = runner._set(E.load_raw(path), sets)
    if sweep:
        raw, variants = E.sweep_variants(raw, path)
        return E.finish(raw, path), variants
    return E.finish(raw, path)
