"""bench/verify/oracle.py: the float64 exact scan of a dense run."""

import pytest


@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_dense_exact_scan_holds_one_file_at_a_time_and_equals_the_one_shot(tmp_path, metric,
                                                                            monkeypatch):
    import types
    from helpers.scan_fixtures import _tiny_corpus
    from bench.verify import agree, oracle as O
    X, Q = _tiny_corpus(tmp_path, n_files=4, rows=9)
    a = types.SimpleNamespace(k=4, modality="dense", metric=metric, corpus=str(tmp_path / "c"),
                              vcol="emb", queries=str(tmp_path / "q.parquet"), qcol="emb",
                              query_rows=None)
    inputs = {"keys": [f"k{i}" for i in range(4)], "offsets": [0, 9, 18, 27, 36], "n": 6,
              "qn": [str(i) for i in range(6)]}
    seen, orig = [], O.exact_topk_f64

    def spy(xq, blocks, k, *aa, **kw):
        seen.append(len(blocks))
        return orig(xq, blocks, k, *aa, **kw)
    monkeypatch.setattr(O, "exact_topk_f64", spy)
    sample = [0, 2, 5]
    got = agree.exact_scan_candidates(a, inputs, sample, 3)
    assert seen == [1, 1, 1]                 # one file per call: never the whole subset at once
    blocks = [types.SimpleNamespace(vectors=X[i * 9:(i + 1) * 9]) for i in range(3)]
    _, I, _, _ = O.oneshot_reference(Q[sample], blocks, 4, "cosine" if metric == "cosine" else "ip")
    assert got == {q: {int(g) for g in I[j] if g >= 0} for j, q in enumerate(sample)}
