"""Verify first-round results for each system, allowing differences only
when explained by float32 rounding.

Default: rescore disputed rows in float64 and verify that each system
returned equally good or better candidates than those it omitted.
This cannot detect errors shared by all systems.

Exhaustive (`checks.exact_scan: true`): additionally compare every rank
against cached float64 ground truth for a reproducible sample of queries
(default 1000, configurable via `checks.queries`).

Both modes check result IDs and filter compliance across all queries
and support dense, sparse, multivector, and filtered search.

Results are written to checks.json and can be regenerated with
`bench verify RUN_DIR`.
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import os
import sys
import time

import numpy as np

from bench.verify import exact as X
from bench.verify import load as L

ULP_EDGES = (8, 16, 32, 64, 128)    # report-only histogram buckets (float32 ULPs)
EXAMPLES = 5
SAMPLE_SEED = 0                     # the exhaustive check's query sample (sample_queries)
SAMPLE_DEFAULT = 1000               # checks.queries when unset
# the float32 bound per modality, as checks.json records it (exact.py has the reasons)
BOUNDS = {
    "dense": "gamma(2d+8) * (||q|| ||x||, or 1 for cosine); pad = -inf",
    "sparse": "ip: gamma(nq+8) ||q|| ||x||; cosine: gamma(2nq+nx+8); nq, nx stored nonzeros; pad = 0",
    "multivector": "(gamma(2d+8) + gamma(T)) * (sum_t ||q_t|| * max_j ||x_j||, or T for cosine); "
                   "T query tokens; pad = -inf",
}
BOUNDS["filtered"] = BOUNDS["dense"]


CheckError = L.CheckError


# ---------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------
def ulps(gap, scale):
    """``gap`` in float32 ULPs at ``scale`` (the query's top score: a row's own score near 0
    would blow the count up)."""
    sp = np.spacing(np.float32(np.where(np.isfinite(scale) & (scale != 0), np.abs(scale), 1.0)))
    return np.asarray(gap, np.float64) / sp.astype(np.float64)


def histogram(u) -> dict:
    u = np.asarray(u, np.float64)
    u = u[np.isfinite(u)]
    out, lo = {}, -np.inf
    for e in ULP_EDGES:
        out[f"<={e}"] = int(((u > lo) & (u <= e)).sum())
        lo = e
    out[f">{ULP_EDGES[-1]}"] = int((u > ULP_EDGES[-1]).sum())
    out["max"] = float(u.max()) if len(u) else 0.0
    return out


def matrices(ids, scores, k, rows=None):
    """(n, k) id and reported-score matrices of one system: -1 / NaN past its list's end; an id
    outside [0, rows) other than -1 becomes -2 (an R1 defect, never looked up)."""
    n = len(ids)
    M = np.full((n, k), -1, np.int64)
    S = np.full((n, k), np.nan)
    for q in range(n):
        a = np.asarray(ids[q][:k], np.int64)
        M[q, :len(a)] = a
        S[q, :len(a)] = np.asarray(scores[q][:len(a)], np.float64)
    if rows is not None:
        M = np.where(((M >= 0) & (M < rows)) | (M == -1), M, -2)
    return M, S


def expect_inexact(exp, name) -> bool:
    """Is system ``name`` listed in checks.expect_inexact (names or fnmatch patterns, e.g.
    "*rerank-released*", matched against the system / sweep variant name)?"""
    return any(fnmatch.fnmatchcase(name, p) for p in (exp.get("checks") or {}).get("expect_inexact") or [])


def sample_queries(n, setting):
    """The exhaustive check's queries: None (every query) for "all" or a size >= n, else
    sorted(default_rng(SAMPLE_SEED).choice(n, size, replace=False)). It depends only on n and
    the setting, never on the systems' results, so the cached reference is reused across runs
    and sweep variants."""
    if setting == "all" or int(setting) >= n:
        return None
    return np.sort(np.random.default_rng(SAMPLE_SEED).choice(n, int(setting), replace=False))


def query_scale(o):
    """Per query: the largest |score| among the finite entries of each row of ``o``."""
    a = np.where(np.isfinite(o), np.abs(o), 0.0)
    return a.max(axis=1) if a.shape[1] else np.zeros(len(o))


# ---------------------------------------------------------------------------------------------
# loading one search's round 1
# ---------------------------------------------------------------------------------------------
def own_result(run_dir, path):
    """A path a leg record names (a nova-bf result <run>/novabf/<tag>/..., its config
    <run>/configs/<tag>.yaml) as the run dir holds it: a run dir copied or synced elsewhere is
    checked on its own files, never on the original's."""
    if not path or "://" in path or path.startswith(os.path.abspath(run_dir) + os.sep):
        return path
    for part in ("novabf", "configs"):
        i = path.rfind(os.sep + part + os.sep)
        if i >= 0:
            mine = os.path.join(run_dir, path[i + 1:])
            if os.path.isfile(mine):
                return mine
    return path


def run_filter_config(exp, run_dir):
    """The filter config the legs used: the run's own copy (configs/filter.yaml, written by the
    first leg). Without it the checks would judge the systems against today's filter."""
    p = os.path.join(run_dir, "configs", "filter.yaml")
    if not os.path.isfile(p):
        raise CheckError(f"{p} missing: the run's own filter config (a partial copy?)")
    return p


def load_search(exp, run_dir, search, recs, skipped, cache_dir=None):
    """-> load_legs of the search's usable round-1 legs (a leg that is not usable is appended
    to ``skipped``)."""
    from bench import timing
    legs = []
    for sy in search["systems"]:
        rec = L.usable(run_dir, recs, timing.tag(search["name"], sy, 1), search, skipped)
        if rec is not None:
            legs.append((sy, rec, "novabf" if rec.get("kind") == "novabf" else "harness"))
    if not legs:
        raise CheckError("no usable round-1 leg")
    return load_legs(exp, run_dir, search, legs, cache_dir=cache_dir)


def load_legs(exp, run_dir, search, legs, own=True, cache_dir=None):
    """-> (a, inputs, scorer, corpus, flt, e2): the results of ``legs`` [(name, record, kind)]
    (load.load_round: nova-bf ids mapped to global rows), the modality's float64 scorer, the corpus
    files, the filter and the experiment as checked (filtered: the run's own filter config).
    ``own``: a nova-bf leg's result and config are the run dir's own copies (a copied / synced
    run dir), and a result outside the run dir is refused; a record that carries its config
    (``_cfg``) is not looked up. ``cache_dir``: the reference cache dir, which also holds the
    corpus's nova-bf uuid table (load.cached_uuid_table); None: the table is built."""
    inside = lambda p: os.path.abspath(p).startswith(os.path.abspath(run_dir) + os.sep)  # noqa: E731
    for sy, rec, kind in legs:
        if kind != "novabf":
            continue
        if own:   # the run dir's own files (a copied / synced run dir)
            rec["_result"] = own_result(run_dir, rec["_result"])
            rec["_config"] = own_result(run_dir, rec.get("_config"))
            r = rec["_result"]
            if "://" not in r and not exp.get("novabf_output") and not inside(r):
                raise CheckError(f"{sy}: nova-bf result {r} is outside the run dir and the run dir "
                                 "has no copy of it (a partial copy?)")
        if "_cfg" not in rec:   # it decides the id mapping
            rec["_cfg"], rec["_cfg_warn"] = novabf_cfg(run_dir, rec.get("_config"), sy)
    nb = [x for x in legs if x[2] == "novabf"]
    id_mode = _id_mode(nb[0][1]["_cfg"]) if nb else "uuid"
    for _, rec, _ in nb[1:]:
        if _id_mode(rec["_cfg"]) != id_mode:
            raise CheckError("nova-bf systems of one search disagree on id_column")
    e2 = copy.deepcopy(exp)
    if exp["modality"] == "filtered":
        e2["filter"] = dict(exp["filter"], config=run_filter_config(exp, run_dir))
    a = L.search_args(e2, search)
    inputs = L.load_round(a, legs, id_mode, cache_dir)
    inputs["config_warnings"] = [r["_cfg_warn"] for _, r, k in legs if k == "novabf" and r.get("_cfg_warn")]
    if exp["modality"] in ("dense", "filtered"):
        scorer = X.Dense(L.dense_queries(a, inputs["n"], inputs["qn"]), a.metric, a.vcol)
    elif exp["modality"] in ("sparse", "multivector"):
        cls = X.Sparse if exp["modality"] == "sparse" else X.MultiVector
        scorer = cls(L.ragged_queries(a, inputs["n"], inputs["qn"]), a.metric, a.vcol)
    else:
        raise CheckError(f"modality {exp['modality']!r}: no scorer")
    fs, paths = L.corpus_paths(a.corpus, len(inputs["keys"]), a.max_files, a.recursive)
    corpus = X.Corpus(fs, paths, inputs["offsets"])
    flt = None
    if exp["modality"] == "filtered":
        job, f = L.filter_setup(e2, search, a, inputs["n"])
        flt = X.Filter(job, f, corpus)
    return a, inputs, scorer, corpus, flt, e2


def novabf_cfg(run_dir, path, system) -> dict:
    """The nova-bf config a leg ran with: the run dir's own copy (<run>/configs/, filtered /
    params legs), else the text manifest.json recorded for a repo config (dense legs name the
    experiment's file, which may since have changed or, on a synced run, live on another machine)."""
    import yaml

    def parse(text, where):
        cfg = yaml.safe_load(text) or {}
        if not isinstance(cfg, dict):
            raise CheckError(f"{system}: nova-bf config {where} is not a YAML mapping")
        return cfg
    if path and os.path.isfile(path) and os.path.abspath(path).startswith(os.path.abspath(run_dir) + os.sep):
        return parse(open(path).read(), path), None
    try:
        files = json.load(open(os.path.join(run_dir, "manifest.json"))).get("files") or {}
    except (OSError, ValueError):
        files = {}
    # the manifest keys a repo config relative to clean_benchmarks: the same file here, or (a run
    # synced from a node) the same path below a directory, never a bare file name
    p = os.path.normpath(str(path or ""))
    hit = [k for k in files if os.path.normpath(os.path.join(L.CB, k)) == p
           or ("/" in k.strip("/") and ".." not in k and p.replace(os.sep, "/").endswith("/" + k.strip("/")))]
    if hit:
        k = max(hit, key=len)
        return parse(files[k], f"manifest.json files[{k!r}]"), None
    if path and os.path.isfile(path):
        return parse(open(path).read(), path), (f"{system}: nova-bf config {path} read as it is now "
                                                "(not in the run dir or manifest.json)")
    raise CheckError(f"{system}: nova-bf config {path} not found, in the run dir or in manifest.json")


def _id_mode(cfg) -> str:
    """How nova-bf names corpus rows: by the corpus `id` column (id_column: id), else by uuid."""
    idc = (cfg.get("corpus") or {}).get("id_column")
    if idc is None:
        return "uuid"
    if idc != "id":
        raise CheckError(f"nova-bf corpus id_column {idc!r}: only `id` is mapped")
    return "corpus"


# ---------------------------------------------------------------------------------------------
# rules shared by both checks
# ---------------------------------------------------------------------------------------------
def defects(ids, k, mapped) -> dict:
    """R1: ids are -1 or real rows, -1 only at the end (any -1 in a nova-bf list = an id that maps
    to no row), no id twice."""
    d, bad = L.list_defects_and_bad(ids, k, mapped)   # bad: per query, any R1 defect
    return {"_bad": bad, "failed": d["failed"], "duplicate_ids": d["queries_with_duplicate_ids"],
            "holes_or_unmapped": d["queries_with_unmapped_entries"],
            "out_of_range": d.get("queries_with_out_of_range_ids", 0), "examples": d["examples"][:EXAMPLES]}


def filter_violations(flt, M):
    """Filtered searches: (query, rank) of every returned row the filter excludes (all rows, not
    only the disputed ones; reads only the filter columns)."""
    qs, js = np.nonzero(M >= 0)
    if not len(qs):
        return qs, js
    ok = flt.allowed(qs, M[qs, js])
    return qs[~ok], js[~ok]


# ---------------------------------------------------------------------------------------------
# default: the disputed rows
# ---------------------------------------------------------------------------------------------
def disputed_check(names, mats, scorer, corpus, flt):
    """{system: verdict} of the disputed-rows rule, and the number of rescored pairs."""
    n = len(next(iter(mats.values()))[0])
    big = np.iinfo(np.int64).max
    srt = {nm: np.sort(np.where(mats[nm][0] >= 0, mats[nm][0], big), axis=1) for nm in names}
    differ = np.zeros(n, bool)
    for nm in names[1:]:
        differ |= (srt[nm] != srt[names[0]]).any(axis=1)
    qd = np.flatnonzero(differ)
    plan, pq_, pg = {}, [], []
    for q in qd:
        L = {nm: set(int(g) for g in mats[nm][0][q] if g >= 0) for nm in names}
        common = set.intersection(*L.values())
        U = set.union(*L.values()) - common
        plan[int(q)] = (L, U)
        pq_ += [int(q)] * len(U)
        pg += sorted(U)
    e, xn = X.score_pairs(scorer, corpus, pq_, pg)
    nan = np.isnan(e)
    if flt is not None and len(pg):
        e = np.where(flt.allowed(pq_, pg), e, -np.inf)
    b = scorer.bound(np.asarray(pq_, np.int64), np.where(np.isnan(xn), 0.0, xn))
    at = {(q, g): i for i, (q, g) in enumerate(zip(pq_, pg))}
    out = {}
    for nm in names:
        M, S = mats[nm]
        bad_q, drop_q, short_q, kept_bad, score_bad, gaps, ex = set(), set(), set(), 0, 0, [], []
        for q, (L, U) in plan.items():
            mq = max(len(v) for v in L.values())
            kept = sorted(L[nm] & U)
            drop = sorted(U - L[nm])
            ki = [at[(q, g)] for g in kept]
            di = [at[(q, g)] for g in drop]
            # a kept row that is not a candidate (filter-excluded or a NaN score)
            nk = [i for i in ki if nan[i] or not np.isfinite(e[i])]
            if nk:
                kept_bad += len(nk)
                bad_q.add(q)
                if len(ex) < EXAMPLES:
                    ex.append({"query": q, "rule": "kept a non-candidate row", "id": pg[nk[0]],
                               "exact": float(e[nk[0]])})
            fk = [i for i in ki if np.isfinite(e[i])]
            lo = min((e[i] + b[i] for i in fk), default=np.inf)
            short = len(L[nm]) < mq
            if short:   # a pad is a kept row scoring the floor (dense / multivector -inf, sparse 0)
                lo = min(lo, scorer.floor)
            fd = [i for i in di if np.isfinite(e[i])]
            hi = max((e[i] - b[i] for i in fd), default=-np.inf)
            if hi > lo:
                bad_q.add(q)
                (short_q if short else drop_q).add(q)
                if len(ex) < EXAMPLES:
                    w = max(fd, key=lambda i: e[i] - b[i])
                    ex.append({"query": q, "rule": "returned fewer rows than another system found"
                               if short else "dropped a clearly better row", "dropped": pg[w],
                               "dropped_exact": float(e[w]),
                               "kept_worst_exact": float(min((e[i] for i in fk), default=-np.inf))})
            if fk and fd:
                g = max(e[i] for i in fd) - min(e[i] for i in fk)
                scale = max(abs(e[i]) for i in fk + fd)
                gaps.append(float(ulps(max(0.0, g), scale)))
            # the reported score of each kept disputed row
            row = {int(g): j for j, g in enumerate(M[q]) if g >= 0}
            for i in fk:
                r = S[q, row[pg[i]]]
                if not abs(r - e[i]) <= b[i]:
                    score_bad += 1
                    bad_q.add(q)
                    if len(ex) < EXAMPLES:
                        ex.append({"query": q, "rule": "reported score off", "id": pg[i],
                                   "reported": float(r), "exact": float(e[i]), "bound": float(b[i])})
        fails = []
        if kept_bad:
            fails.append(f"{kept_bad} kept rows are not candidates (filter-excluded or NaN)")
        if drop_q:
            fails.append(f"{len(drop_q)} queries kept a row clearly worse than one it dropped "
                         "(beyond the float32 bound)")
        if short_q:
            fails.append(f"{len(short_q)} queries returned fewer rows than another system found")
        if score_bad:
            fails.append(f"{score_bad} reported scores off their float64 score by more than the float32 bound")
        bq = np.zeros(n, bool)
        bq[list(bad_q)] = True
        out[nm] = {"fails": fails, "_bad": bq, "examples": ex, "ulp_gaps": histogram(gaps)}
    return out, len(pg), len(qd)


# ---------------------------------------------------------------------------------------------
# flagged: the exhaustive reference
# ---------------------------------------------------------------------------------------------
def exhaustive_check(names, mats, k, scorer, corpus, flt, ref):
    """{system: verdict} of rules R2-R4 against the reference ``ref`` (build_reference: D, I, N
    top-K by exact score; P, PI top-K by exact score minus bound; K > k).

    The set rule needs the best (exact - bound) over the rows a system dropped. A system holds at
    most k rows, so that row is the first entry of P / PI that is not in its list."""
    D, I, N, P, PI = (ref[x] for x in ("D", "I", "N", "P", "PI"))
    n = len(D)
    stride = corpus.rows + 1
    rk = (np.arange(n)[:, None] * stride + I)[I >= 0]
    rD, rN = D[I >= 0], N[I >= 0]
    o_ = np.argsort(rk)
    rk, rD, rN = rk[o_], rD[o_], rN[o_]
    o = D[:, :k]
    ob = np.where(I[:, :k] >= 0, scorer.bound(np.arange(n)[:, None], np.where(np.isnan(N[:, :k]), 0.0, N[:, :k])), 0.0)
    scale = query_scale(o)
    n_ref = ((I[:, :k] >= 0) & np.isfinite(o)).sum(axis=1)   # rows a full list must hold
    pkey = np.arange(n)[:, None] * stride + PI
    out = {}
    for nm in names:
        M, S = mats[nm]
        real = M >= 0
        e = np.full(M.shape, -np.inf)
        xn = np.zeros(M.shape)
        key = np.arange(n)[:, None] * stride + np.maximum(M, 0)
        pos = np.clip(np.searchsorted(rk, key), 0, max(0, len(rk) - 1))
        found = real & (rk[pos] == key) if len(rk) else np.zeros(M.shape, bool)
        e[found], xn[found] = rD[pos[found]], rN[pos[found]]
        miss = real & ~found                     # outside the reference's K: score from the corpus
        qs, js = np.nonzero(miss)
        rescored = len(qs)
        if rescored:
            em, xm = X.score_pairs(scorer, corpus, qs, M[qs, js])
            if flt is not None:
                em = np.where(flt.allowed(qs, M[qs, js]), em, -np.inf)
            e[qs, js], xn[qs, js] = em, xm
        xn = np.where(np.isnan(xn), 0.0, xn)
        b = np.where(real, scorer.bound(np.arange(n)[:, None], xn), 0.0)
        kept = real & np.isfinite(e)
        r2 = real & ~np.isfinite(e)                                   # R2: not a candidate
        # R3 set: every row the system dropped must be a tie with, or worse than, every row it
        # kept (the disputed rule, with every corpus row as the other side). A list shorter than
        # the reference's finite rows has a pad standing for the floor (-inf: it loses to any
        # real row; sparse 0: it loses to a row clearly above 0).
        dropped = (PI >= 0) & np.isfinite(P) & ~np.isin(pkey, (np.arange(n)[:, None] * stride + M)[real])
        hi = np.where(dropped, P, -np.inf).max(axis=1)
        lo = np.where(kept, e + b, np.inf).min(axis=1)
        lo = np.where(real.sum(axis=1) < n_ref, np.minimum(lo, scorer.floor), lo)
        r3 = hi > lo
        # R3 order: no row ranked lower beats a higher-ranked one by more than their bounds
        v = np.where(kept, e - b, -np.inf)
        below = np.concatenate([np.maximum.accumulate(v[:, ::-1], axis=1)[:, ::-1][:, 1:],
                                np.full((n, 1), -np.inf)], axis=1)
        r3o = kept & (below > e + b)
        r4 = kept & ~(np.abs(S - e) <= b)                             # R4: reported score
        bad = r2 | r3o | r4 | r3[:, None]
        ex = []
        for q in np.flatnonzero(r3)[:EXAMPLES]:
            j = int(np.argmax(np.where(dropped[q], P[q], -np.inf)))
            ex.append({"query": int(q), "rule": "R3 dropped a clearly better row", "dropped": int(PI[q, j]),
                       "dropped_exact_minus_bound": float(P[q, j]),
                       "kept_worst_exact": float(np.where(kept[q], e[q], np.inf).min())
                       if kept[q].any() else None, "short_list": bool(real[q].sum() < n_ref[q])})
        for q, j in zip(*np.nonzero(r2 | r3o | r4)):
            if len(ex) >= EXAMPLES:
                break
            rule = "R2 not a candidate" if r2[q, j] else ("R3 order" if r3o[q, j] else "R4 reported score")
            ex.append({"query": int(q), "rank": int(j), "id": int(M[q, j]), "exact": float(e[q, j]),
                       "reported": float(S[q, j]), "bound": float(b[q, j]), "rule": rule})
        # report only: the rank-by-rank gap to the reference (not a rule: a shifted list can have
        # small gaps, and with ip an honest list can exceed the bound at one rank)
        both = np.isfinite(e) & np.isfinite(o)
        gaps = ulps(np.abs(e - o), np.broadcast_to(scale[:, None], o.shape))[both]
        fails = []
        if r2.any():
            fails.append(f"R2: {int(r2.sum())} returned rows are not candidates (filter-excluded or NaN)")
        if r3.any():
            fails.append(f"R3: {int(r3.sum())} queries dropped a row clearly better than one they kept "
                         "(beyond the float32 bound), or returned too few rows")
        if r3o.any():
            fails.append(f"R3: {int(r3o.any(axis=1).sum())} queries out of order beyond the float32 bound")
        if r4.any():
            fails.append(f"R4: {int(r4.sum())} reported scores off their float64 score by more than the bound")
        rel = np.abs(e - o)[both] / np.maximum((b + ob)[both], 1e-300)
        out[nm] = {"fails": fails, "_bad": bad.any(axis=1), "examples": ex,
                   "ulp_gaps": histogram(gaps), "rescored_outside_reference": int(rescored),
                   "max_gap_over_bound": float(rel.max()) if rel.size else 0.0,
                   "max_score_error": float(np.abs(S - e)[real & np.isfinite(e)].max())
                   if (real & np.isfinite(e)).any() else 0.0}
    return out


# ---------------------------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------------------------
def _filter_id(exp, search):
    if exp["modality"] != "filtered":
        return None
    p = exp["filter"]["config"]
    return [X._sha256(p) if os.path.isfile(p) else p, search.get("filter_search") or search["name"]]


def check_search(exp, run_dir, search, recs, skipped, exact_scan, cache_dir):
    """One search's document: the verdict of every usable round-1 leg."""
    sec, t = {}, time.time()
    loaded = load_search(exp, run_dir, search, recs, skipped, cache_dir)
    sec["load"] = time.time() - t
    doc = judge_search(exp, search, loaded, exact_scan, cache_dir, sec)
    doc["warnings"] = loaded[1]["config_warnings"]
    return doc


def judge_search(exp, search, loaded, exact_scan, cache_dir, sec):
    """The rules on one search's loaded results (load_legs): R1 list defects, the filter, then
    the default disputed-rows check or (``exact_scan``) the exhaustive one."""
    a, inputs, scorer, corpus, flt, e2 = loaded
    k = int(exp["search"]["k"])
    names = list(inputs["systems"])
    mats = {nm: matrices(*inputs["systems"][nm], k, corpus.rows) for nm in names}
    res = {nm: {"fails": []} for nm in names}
    t = time.time()
    for nm in names:
        d = defects(inputs["systems"][nm][0], k, inputs["mapped"].get(nm, False))
        r1q = d.pop("_bad")
        res[nm]["defects"] = d
        res[nm]["_pre"] = r1q
        if d["failed"]:
            res[nm]["fails"].append("R1: list defects (duplicate ids, holes / unmapped ids, "
                                    "or out-of-range ids)")
        if flt is not None:
            vq, vj = filter_violations(flt, mats[nm][0])
            res[nm]["filter_violations"] = int(len(vq))
            r1q[vq] = True
            if len(vq):
                res[nm]["fails"].append(f"{len(vq)} returned rows the filter excludes")
    sec["defects_filter"] = time.time() - t
    doc = {"check": "exhaustive" if exact_scan else "disputed", "queries": inputs["n"], "k": k,
           "rows": corpus.rows, "files": len(corpus.paths), "systems": res,
           "run_problems": scorer.run_problems(corpus)}   # sparse: negative weights
    t = time.time()
    n = inputs["n"]
    # the disputed-rows rule: the default check, and part of the exhaustive one (every query)
    per, pairs, nq = disputed_check(names, mats, scorer, corpus, flt) if len(names) > 1 \
        else ({nm: {"fails": [], "_bad": np.zeros(n, bool), "examples": [],
                    "ulp_gaps": histogram([])} for nm in names}, 0, 0)
    doc["disputed"] = {"queries": nq, "pairs_rescored": pairs}
    if exact_scan:
        sec["disputed"] = time.time() - t
        t = time.time()
        K = k + X.SPARE
        setting = (exp.get("checks") or {}).get("queries", SAMPLE_DEFAULT)
        qs = sample_queries(n, setting)              # None: every query
        m = n if qs is None else len(qs)
        doc["sample"] = {"queries": m, "of": n, "seed": SAMPLE_SEED, "setting": setting}
        key = X.reference_key(corpus, a.queries, a.query_rows, a.vcol, a.qcol, a.metric,
                              exp["modality"], _filter_id(e2, search), K,
                              "all" if qs is None else {"seed": SAMPLE_SEED, "size": m, "n": n})
        sc, fl = (scorer, flt) if qs is None else (scorer.subset(qs), flt and flt.subset(qs))
        ref, hit = X.cached_reference(cache_dir, key, lambda: X.build_reference(sc, corpus, K, fl),
                                      (m, K))
        I = ref["I"]
        sec["reference"] = time.time() - t
        t = time.time()
        doc["reference"] = {"key": key, "cached": hit, "K": K, "cache_dir": cache_dir,
                            "device": ref.pop("device"),
                            "short_queries": int((I[:, :k] < 0).any(axis=1).sum())}
        sub = mats if qs is None else {nm: (mats[nm][0][qs], mats[nm][1][qs]) for nm in names}
        exh = exhaustive_check(names, sub, k, sc, corpus, fl, ref)
        for nm in names:   # the two parts: the exhaustive fields at the top, the disputed ones below
            d, e = per[nm], exh[nm]
            if qs is not None:   # back to the search's query numbers
                bad = np.zeros(n, bool)
                bad[qs] = e["_bad"]
                e["_bad"] = bad
                for x in e["examples"]:
                    x["query"] = int(qs[x["query"]])
            e["failed_parts"] = [p for p, f in (("disputed", d["fails"]), ("exhaustive", e["fails"])) if f]
            e["fails"] = ([f"disputed rows (every query): {f}" for f in d["fails"]]
                          + [f if qs is None else f"exhaustive ({m} of {n} queries): {f}" for f in e["fails"]])
            e["_bad"] = e["_bad"] | d.pop("_bad")
            e["disputed"] = {"fails": d["fails"], "examples": d["examples"], "ulp_gaps": d["ulp_gaps"]}
        per = exh
        doc["blind_spot"] = None if qs is None else (
            f"outside the {m}-query sample (of {n}): a mistake every system makes together (a row "
            "all missed, the same wrong row in every list, the order inside a list, the reported "
            "score of a row every system returned); disagreements between systems are still "
            "judged on every query (checks.queries: all covers every query)")
    else:
        doc["blind_spot"] = ("not seen: a row every system missed, the same wrong row in every list, "
                             "the order inside a list, and the reported score of a row every "
                             "system returned (checks.exact_scan: true sees all of these)")
        if len(names) == 1:
            doc["blind_spot"] = "one system: nothing to compare (checks.exact_scan: true checks it)"
    sec["judge"] = time.time() - t
    for nm in names:
        res[nm]["fails"] += per[nm].pop("fails")
        res[nm]["queries_failed"] = int((res[nm].pop("_pre") | per[nm].pop("_bad")).sum())
        res[nm].update(per[nm])
        res[nm]["ok"] = not res[nm]["fails"]
        if expect_inexact(exp, nm):
            res[nm]["expect_inexact"] = True
    doc["seconds"] = sec
    return doc


def e2e_check(exp, recs, run_dir):
    """The cold e2e leg's result vs round 1 (kept from the old checks; no corpus read)."""
    from bench import timing
    if not exp.get("e2e"):
        return None, []
    sy = exp["e2e"]["system"]
    s0 = exp["searches"][0]
    e2e, first = recs.get(f"e2e__{sy}"), recs.get(timing.tag(s0["name"], sy, 1))
    if not (e2e and e2e.get("ok")):   # configured, so it ran: missing or failed is a failure
        return "ERROR", [f"e2e leg e2e__{sy} " + ("missing" if not e2e else "failed")]
    if not (first and first.get("ok")):
        return None, []   # round 1 itself is missing / failed: reported as a skipped leg
    by_id = (exp["e2e"].get("tiebreak") or {}).get("mode") == "id"
    cmp = L.same_scores if by_id else L.same_table
    try:
        same = cmp(own_result(run_dir, L.novabf_result(e2e, s0)),
                   own_result(run_dir, L.novabf_result(first, s0)))
    except (CheckError, OSError) as exc:
        return "ERROR", [f"e2e: {exc}"]
    if same:
        return ("SCORES IDENTICAL" if by_id else "IDENTICAL"), []
    return "DIFFERENT", [f"e2e vs {s0['name']} r1 result: DIFFERENT"]


def run_inputs(run_dir) -> dict:
    """{file: size + sha256} of what the checks read from the run: its own files (keyed relative
    to the run dir, so a copy matches), nova-bf results and configs as the run dir holds them."""
    keys = set(L.input_files(run_dir))
    for root, _, files in os.walk(os.path.join(run_dir, "configs")):   # nested dirs keep their path
        keys |= {os.path.relpath(os.path.join(root, f), run_dir).replace(os.sep, "/")
                 for f in files if f.endswith(".yaml")}
    out = {}
    for k in sorted(keys):
        p = own_result(run_dir, k)
        rel = os.path.relpath(p, run_dir) if "://" not in p and os.path.isabs(p) and \
            os.path.abspath(p).startswith(os.path.abspath(run_dir) + os.sep) else p
        out[rel] = L.file_id(run_dir, p)
    return out


def input_problems(run_dir, doc) -> list[str]:
    """Why a checks.json no longer describes the run dir: no `inputs` record, a recorded file
    changed or gone, or a file the checks would now read that it did not record (run_inputs
    re-taken; checks.json keys hold private paths in placeholder form, bench/paths.py)."""
    rec = doc.get("inputs")
    if not isinstance(rec, dict) or not rec:
        return ["checks.json records no inputs (written before input hashing, or by hand): "
                "re-run the checks (bench verify RUN_DIR)"]
    from bench import paths
    now = {paths.mask(k): v for k, v in run_inputs(run_dir).items()}
    out = []
    for k in sorted(set(rec) | set(now)):
        if k not in rec:
            out.append(f"{k}: not read by the checks (added after them)")
            continue
        cur = now[k] if k in now else L.file_id(run_dir, paths.expand(k))
        if cur.get("missing"):
            out.append(f"{k}: missing (the checks read it)")
        elif cur != rec[k]:
            out.append(f"{k}: changed since the checks read it")
    return out


def run(run_dir, exp, exact_scan=None, cache_dir=None) -> dict:
    """-> the checks.json document; `ok` false = a failed check."""
    from bench import runner
    t_all = time.time()
    if exact_scan is None:
        exact_scan = bool((exp.get("checks") or {}).get("exact_scan"))
    cache_dir = cache_dir or os.path.join(runner.RESULTS, "oracle_cache")
    doc = {"ok": True, "problems": [], "warnings": [], "skipped": [],
           "rule": {"bound": BOUNDS.get(exp["modality"], "?") + "; gamma(m) = m u / (1 - m u), u = 2^-24",
                    "round": 1, "check": "exhaustive" if exact_scan else "disputed"},
           "searches": {}, "inputs": run_inputs(run_dir)}
    fc = os.path.join(run_dir, "configs", "filter.yaml")
    if os.path.isfile(fc):
        doc["inputs"]["configs/filter.yaml"] = L.file_id(run_dir, "configs/filter.yaml")
    recs = L.records(run_dir)
    L.CACHE.clear()
    try:
        with L.threads(L.check_threads(exp)):
            for s in exp["searches"]:
                try:
                    d = check_search(exp, run_dir, s, recs, doc["skipped"], exact_scan, cache_dir)
                except Exception as exc:  # noqa: BLE001  a check that cannot run is a failed check
                    doc["searches"][s["name"]] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    doc["problems"].append(f"{s['name']}: could not check: {type(exc).__name__}: {exc}")
                    continue
                doc["searches"][s["name"]] = d
                doc["warnings"] += [f"{s['name']}: {w}" for w in d.pop("warnings")]
                doc["problems"] += [f"{s['name']}: {p}" for p in d["run_problems"]]
                if len(d["systems"]) == 1 and d["check"] == "disputed":
                    doc["warnings"].append(f"{s['name']}: one system, so the default check compared "
                                           "nothing (checks.exact_scan: true checks it)")
                for nm, v in d["systems"].items():
                    if v["ok"]:
                        continue
                    msg = f"{s['name']} {nm}: " + "; ".join(v["fails"])
                    inexact = expect_inexact(exp, nm)
                    (doc["warnings"] if inexact else doc["problems"]).append(
                        msg + (" (expect_inexact: reported, not failed)" if inexact else ""))
    finally:
        L.CACHE.clear()
    # a leg the checks could not read is a check that cannot run: a failure, never silent
    doc["problems"] += [f"leg {x['leg']} not checked: {x['why']}" for x in doc["skipped"]]
    doc["e2e"], p = e2e_check(exp, recs, run_dir)
    doc["problems"] += p
    doc["ok"] = not doc["problems"]
    doc["seconds"] = time.time() - t_all
    return doc


def failed_doc(run_dir, why) -> dict:
    """A failing checks.json for checks that could not run (``why``), with the inputs recorded
    when they can be (so it is not also reported stale); without them when hashing fails too."""
    try:
        inputs = run_inputs(run_dir)
    except Exception:  # noqa: BLE001
        inputs = {}
    return {"ok": False, "problems": [f"checks could not run: {why}"], "warnings": [],
            "skipped": [], "searches": {}, "e2e": None, "inputs": inputs}


def run_or_fail(run_dir, exp, exact_scan=None, cache_dir=None) -> tuple[dict, bool]:
    """(checks.json document, crashed): run() or, when it raises anything (a modality it cannot
    read, expired S3 credentials, an unreadable leg file), a failed check saying why. bench run,
    bench sweep and bench verify all use this, so none ends without a verdict."""
    try:
        return run(run_dir, exp, exact_scan, cache_dir), False
    except Exception as exc:  # noqa: BLE001  a check that cannot run is a failed check
        return failed_doc(run_dir, f"{type(exc).__name__}: {exc}"), True


def could_not_run(run_dir, exc) -> dict:
    """The checks.json of a run whose checks could not start (CheckError, e.g. an unreadable
    experiment): a failed check, with the inputs recorded so it is not also reported stale."""
    return {"ok": False, "problems": [f"checks could not run: {exc}"], "warnings": [],
            "skipped": [], "searches": {}, "e2e": None, "inputs": run_inputs(run_dir)}


def write(run_dir, doc) -> str:
    """checks.json in ``run_dir`` (private paths masked, as the summaries are)."""
    from bench import paths
    out = os.path.join(run_dir, "checks.json")
    with open(out, "w") as f:
        json.dump(paths.mask(doc), f, indent=1, sort_keys=True, default=float)
    return out


def compare_lists(exp, run_dir, search, legs) -> dict:
    """The default disputed-rows rule (with R1 and the filter) on exactly ``legs`` [(name,
    record, kind)] of the search named ``search``, as compared by hand (s8/compare_s8.py): a
    harness record holds `_topk` (its saved top-k), a nova-bf one `_result` (its result
    parquet) and `_config` (its config path) or `_cfg` (the config itself). Files are read
    where the records say, not looked up in the run dir. -> {"tie_equivalent": every leg
    passed, "fails": ["name: why", ...]}."""
    s = next(x for x in exp["searches"] if x["name"] == search)
    L.CACHE.clear()
    try:
        with L.threads(L.check_threads(exp)):
            loaded = load_legs(exp, run_dir, s, [(n, dict(r), k) for n, r, k in legs], own=False)
            d = judge_search(exp, s, loaded, False, None, {})
            fails = [f"{nm}: {f}" for nm, v in d["systems"].items() for f in v["fails"]]
            if len(legs) == 2:
                fails += rank_differences(exp, loaded)
    finally:
        L.CACHE.clear()
    return {"tie_equivalent": not fails, "fails": fails}


def rank_differences(exp, loaded) -> list[str]:
    """Two lists compared rank by rank (s8's "tie-equivalent", which also covers order and the
    reported scores): where their ids differ at a rank, the two rows must score within their
    float32 bounds of each other; where the ids agree but the reported scores differ, each
    reported score must be within its bound of the exact one. Only those rows are rescored, so
    two identical outputs cost nothing."""
    a, inputs, scorer, corpus, flt, _ = loaded
    k = int(exp["search"]["k"])
    (na, (MA, SA)), (nb, (MB, SB)) = [(nm, matrices(*inputs["systems"][nm], k, corpus.rows))
                                      for nm in inputs["systems"]]
    both = (MA >= 0) & (MB >= 0)
    moved = both & (MA != MB)
    rescaled = both & (MA == MB) & ~(np.abs(SA - SB) <= 1e-6 * np.maximum(np.abs(SA), np.abs(SB)) + 1e-7)
    qs, js = np.nonzero(moved | rescaled)
    if not len(qs):
        return []
    pq_ = np.concatenate([qs, qs])
    pg = np.concatenate([MA[qs, js], MB[qs, js]])
    e, aux = X.score_pairs(scorer, corpus, pq_, pg)
    if flt is not None:
        e = np.where(flt.allowed(pq_, pg), e, -np.inf)
    b = scorer.bound(pq_, np.where(np.isnan(aux), 0.0, aux))
    m = len(qs)
    eA, eB, bA, bB = e[:m], e[m:], b[:m], b[m:]
    out = []
    mv = moved[qs, js]
    bad = mv & ~(np.abs(eA - eB) <= bA + bB)
    if bad.any():
        out.append(f"{na} vs {nb}: {int(bad.sum())} ranks hold rows whose exact scores differ by "
                   "more than their float32 bounds (order or a different row)")
    for nm, S, ee, bb in ((na, SA, eA, bA), (nb, SB, eB, bB)):
        off = ~(np.abs(S[qs, js] - ee) <= bb)
        if off.any():
            out.append(f"{nm}: {int(off.sum())} reported scores off their float64 score by more "
                       "than the float32 bound")
    return out


def refresh_reports(run_dir, exp) -> None:
    """summary.md (and a sweep's sweep_table.csv / .md, which --pick reads) rebuilt from the new
    checks.json, so no report keeps the verdict of the checks it replaced."""
    from bench.report import summary as S
    rows, problems, excluded, e2e = S.summarize(run_dir)
    S.write(run_dir, rows, problems, excluded, e2e)
    sw = os.path.join(run_dir, "sweep.json")
    if os.path.isfile(sw):
        from bench import runner
        runner.sweep_table(run_dir, exp, json.load(open(sw))["variants"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench verify", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--exact-scan", action="store_true", default=None,
                    help="the exhaustive check (default: the experiment's checks.exact_scan)")
    ap.add_argument("--cache", default=None, help="reference cache dir (default results/oracle_cache)")
    a = ap.parse_args(argv)
    exp = json.load(open(os.path.join(a.run_dir, "experiment.json")))
    from bench import paths
    paths.use(exp)   # the experiment's ${READ} / ${WRITE} section, as bench run resolves them
    out = os.path.join(a.run_dir, "checks.json")
    if os.path.exists(out):
        os.remove(out)   # never leave an older verdict behind a run that cannot be checked
    try:
        doc, crashed = run_or_fail(a.run_dir, exp, a.exact_scan, a.cache)
    except BaseException:   # Ctrl-C: the reports must not keep the replaced verdict either
        write(a.run_dir, failed_doc(a.run_dir, "interrupted"))
        refresh_reports(a.run_dir, exp)
        raise
    write(a.run_dir, doc)
    refresh_reports(a.run_dir, exp)
    if crashed:
        print(f"bench verify: {doc['problems'][0]}", file=sys.stderr)
        return 2
    for s, d in doc["searches"].items():
        if "systems" not in d:
            print(f"{s}: ERROR {d.get('error')}")
            continue
        smp = d.get("sample")
        print(f"{s}: {d['check']} check, {d['queries']} queries"
              + (f" (exhaustive rules on {smp['queries']}, seed {smp['seed']})" if smp else "")
              + (f", reference on {d['reference']['device']}" if d.get("reference") else "")
              + f", {d['rows']} rows, {sum(d['seconds'].values()):.1f} s")
        for nm, v in d["systems"].items():
            print(f"  {nm}: {'ok' if v['ok'] else 'FAIL'}"
                  + ("" if v["ok"] else "  " + "; ".join(v["fails"]))
                  + f"  (ulp gaps max {v['ulp_gaps']['max']:.1f})")
    for p in doc["problems"]:
        print(f"problem: {p}")
    for w in doc["warnings"]:
        print(f"warning: {w}")
    print(f"bench verify: {'PASS' if doc['ok'] else 'FAIL'} in {doc['seconds']:.1f} s "
          f"({out})")
    return 0 if doc["ok"] else 1
