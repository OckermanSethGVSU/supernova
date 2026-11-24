"""Harness rows for the attribution tests (bench/leg.py, bench/report/summary.py)."""




def _row(**kw):
    base = {"source_stall_s": 60.0, "engine_call_s": 38.5, "merge_s": 1.5, "other_s": 0.0,
            "h2d_exposed_s": 0.1, "widen_s": 0.0, "normalize_s": None, "search_s": 38.2,
            "lib_s": 38.0, "ids_s": 0.05, "d2h_s": 0.15, "engine_gap_s": 0.05, "gpu_idle_s": 62.0,
            "build_s": 0.0}
    base.update(kw)
    return base


# distinct values, so a part computed from the wrong column (or with time moved into or out of
# it, the sum unchanged) cannot pass
ROW = {"source_stall_s": 11.0, "engine_call_s": 23.0, "merge_s": 3.0, "other_s": 0.7,
       "h2d_exposed_s": 0.3, "h2d_s": 0.9, "widen_s": 0.13, "normalize_s": 0.17,
       "search_s": 19.0, "lib_s": 17.0, "ids_s": 0.29, "d2h_s": 0.41, "engine_gap_s": 0.05,
       "gpu_idle_s": 0.0, "build_s": 0.0}
ROW["gpu_idle_s"] = 37.7 - (19.0 + 0.13 + 0.17)   # total_s - busy (widen + normalize + search)
CLOCK = 37.7


P_ATTR_ROW = {"total_s": 1.0, "source_stall_s": 0.2, "engine_call_s": 0.5, "merge_s": 0.2,
              "other_s": 0.1, "engine_gap_s": 0.0, "lib_s": None, "search_s": 0.0,
              "gpu_idle_s": None, "h2d_overlapped": 0.0}
