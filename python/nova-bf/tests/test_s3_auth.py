"""Per-location S3 auth (`s3:` on corpus/queries/output).

Offline except `test_live_public_bucket_anonymous`, which reads the public
fineweb-gte-with-payloads bucket and runs only with NOVA_BF_LIVE_S3=1.
"""

import os
import textwrap

import pytest
import pyarrow.fs as pafs
from pydantic import ValidationError

from nova_bf import io as nio
from nova_bf.config import S3AccessConfig, load_config
from nova_bf.io import Store

KEY, SECRET, TOKEN = "AKIAEXAMPLEKEYID0000", "example-secret-value", "example-session-token"


@pytest.fixture(autouse=True)
def _private_native_cache(monkeypatch):
    """Each test gets its own native-client cache, so the clients it builds
    (each with a tokio runtime and worker threads) are dropped when it ends
    instead of staying in the process-wide cache for the rest of the run,
    where their address space counts against later tests' `ulimit -v`."""
    import gc
    monkeypatch.setitem(nio._NATIVE, "clients", {})
    monkeypatch.setitem(nio._NATIVE, "orphans", [])
    yield
    monkeypatch.undo()
    gc.collect()


# ---- config ---------------------------------------------------------------

def test_modes_accepted():
    assert S3AccessConfig(anonymous=True).mode == "anonymous"
    s = S3AccessConfig(access_key_id=KEY, secret_access_key=SECRET)
    assert s.mode == "static" and s.session_token is None
    s = S3AccessConfig(access_key_id=KEY, secret_access_key=SECRET, session_token=TOKEN)
    assert s.session_token.get_secret_value() == TOKEN


@pytest.mark.parametrize("kw", [
    {},                                                    # empty block: say what you mean
    {"anonymous": False},
    {"access_key_id": KEY},                                # key without secret
    {"secret_access_key": SECRET},                         # secret without key
    {"session_token": TOKEN},                              # token alone
    {"anonymous": True, "access_key_id": KEY, "secret_access_key": SECRET},
    {"anonymous": True, "session_token": TOKEN},
    {"anonymous": True, "profile": "x"},                   # unknown field
])
def test_invalid_blocks_rejected(kw):
    with pytest.raises(ValidationError):
        S3AccessConfig(**kw)


@pytest.mark.parametrize("kw", [
    {"access_key_id": "", "secret_access_key": SECRET},
    {"access_key_id": KEY, "secret_access_key": "   "},
    {"access_key_id": KEY, "secret_access_key": SECRET, "session_token": ""},
])
def test_blank_credentials_rejected(kw):
    with pytest.raises(ValidationError, match="empty"):
        S3AccessConfig(**kw)


def test_dumped_config_round_trips(tmp_path, monkeypatch):
    """model_dump writes `s3: null` for sections without a block; that must
    reload as the default chain (tools round-trip configs this way)."""
    import yaml
    from nova_bf.config import BruteForceConfig
    cfg = BruteForceConfig(
        corpus={"path": "s3://b/c/", "s3": {"anonymous": True}},
        queries={"path": "/tmp/q.parquet"}, output={"path": "s3://b/o/"},
        searches=[{"k": 10, "metric": "dot"}])
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    back = load_config(str(p))
    assert back.corpus.s3.anonymous and back.queries.s3 is None and back.output.s3 is None


@pytest.mark.parametrize("bad_yaml", [
    # misspelled key next to real secrets
    "  s3:\n    access_key_id: ${T_KEY}\n    secret_acess_key: ${T_SECRET}\n",
    # both modes
    "  s3:\n    anonymous: true\n    access_key_id: ${T_KEY}\n"
    "    secret_access_key: ${T_SECRET}\n    session_token: ${T_TOKEN}\n",
    # an unrelated queries error with the block present (parent echoes input)
    "  payload_fields: [query_id]\n  s3:\n    access_key_id: ${T_KEY}\n"
    "    secret_access_key: ${T_SECRET}\n    session_token: ${T_TOKEN}\n",
])
def test_validation_errors_never_echo_secrets(tmp_path, monkeypatch, bad_yaml):
    monkeypatch.setenv("T_KEY", KEY)
    monkeypatch.setenv("T_SECRET", SECRET)
    monkeypatch.setenv("T_TOKEN", TOKEN)
    p = tmp_path / "c.yaml"
    p.write_text("corpus:\n  path: s3://b/c/\nqueries:\n  path: /tmp/q.parquet\n"
                 + bad_yaml + "output:\n  path: s3://b/o/\n"
                 "searches:\n  - {k: 10, metric: dot}\n")
    with pytest.raises(ValidationError) as e:
        load_config(str(p))
    text = str(e.value) + repr(e.value)
    for secret in (KEY, SECRET, TOKEN, SECRET[-12:], TOKEN[-12:]):
        assert secret not in text, text


def test_repr_and_dump_hide_credentials():
    s = S3AccessConfig(access_key_id=KEY, secret_access_key=SECRET, session_token=TOKEN)
    for text in (repr(s), str(s), str(s.model_dump()), s.model_dump_json()):
        for secret in (KEY, SECRET, TOKEN):
            assert secret not in text


def test_yaml_per_location_with_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("T_KEY", KEY)
    monkeypatch.setenv("T_SECRET", SECRET)
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""
        corpus:
          path: s3://public-bucket/corpus/?region=us-east-1
          s3: {anonymous: true}
        queries:
          path: /tmp/q.parquet
        output:
          path: s3://my-bucket/out/
          s3:
            access_key_id: ${T_KEY}
            secret_access_key: ${T_SECRET}
        searches:
          - {k: 10, metric: dot}
    """))
    cfg = load_config(str(p))
    assert cfg.corpus.s3.anonymous
    assert cfg.queries.s3 is None
    assert cfg.output.s3.secret_access_key.get_secret_value() == SECRET


# ---- Store / pyarrow ------------------------------------------------------

def _fs_kwargs(fs):
    # S3FileSystem pickles via (_reconstruct, (kwargs,)): its resolved options.
    return fs.__reduce__()[1][0]


def test_store_anonymous_builds_unsigned_fs():
    s = Store("s3://bucket/some/prefix/?region=eu-west-2", s3=S3AccessConfig(anonymous=True))
    assert isinstance(s.fs, pafs.S3FileSystem)
    assert s.root == "bucket/some/prefix"
    kw = _fs_kwargs(s.fs)
    assert kw["anonymous"] is True and kw["region"] == "eu-west-2"
    assert s._native_ok and s._native_region == "eu-west-2"


def test_store_static_keys_override_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAENVENVENVENV0000")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "env-token")
    s = Store("s3://bucket/p?region=us-east-1",
              s3=S3AccessConfig(access_key_id=KEY, secret_access_key=SECRET))
    kw = _fs_kwargs(s.fs)
    assert kw["access_key"] == KEY and kw["secret_key"] == SECRET
    assert not kw.get("session_token")
    assert not kw["anonymous"]


def test_store_root_matches_from_uri():
    # Same root string as the default (from_uri) path, so keys/ids don't change.
    uri = "s3://bucket/a/b/?region=us-east-1"
    assert Store(uri, s3=S3AccessConfig(anonymous=True)).root == pafs.FileSystem.from_uri(uri)[1]


def test_store_rejects_s3_block_on_local_path(tmp_path):
    with pytest.raises(ValueError, match="not an s3://"):
        Store(str(tmp_path), s3=S3AccessConfig(anonymous=True))


def test_store_rejects_unknown_query_params():
    # pyarrow's own URI parser refuses it (ArrowInvalid is a ValueError).
    with pytest.raises(ValueError):
        Store("s3://bucket/p?region=us-east-1&bogus=1", s3=S3AccessConfig(anonymous=True))


@pytest.mark.parametrize("uri", [
    "s3://b/a%20b/c%2Fd?region=us-east-1",       # percent-encoding is decoded
    "s3://b/a%2Bb/?region=us-east-1",
    "s3://b:443/x/?region=us-east-1",            # port stripped
    "s3://b/x?region=us-east-1&region=eu-west-1",  # first region wins
    "s3://b?region=us-east-1",
    "s3://b/?region=us-east-1",
    "s3://b/x//y/?region=us-east-1",
])
def test_store_root_and_region_match_default_path(uri):
    want_fs, want_root = pafs.FileSystem.from_uri(uri)
    for auth in (S3AccessConfig(anonymous=True),
                 S3AccessConfig(access_key_id=KEY, secret_access_key=SECRET)):
        s = Store(uri, s3=auth)
        assert s.root == want_root
        assert s.fs.region == want_fs.region == s._native_region


def test_store_keeps_endpoint_env(monkeypatch):
    # A custom endpoint (MinIO/localstack) must not silently become real AWS.
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9")
    s = Store("s3://only-on-minio/x", s3=S3AccessConfig(anonymous=True))
    assert _fs_kwargs(s.fs)["endpoint_override"] == "http://127.0.0.1:9"
    assert not s._native_ok  # native reader stays off under endpoint env, as before


def test_store_rejects_uri_credentials_with_block():
    with pytest.raises(ValueError, match="conflict") as e:
        Store(f"s3://{KEY}:{SECRET}@b/x?region=us-east-1", s3=S3AccessConfig(anonymous=True))
    assert SECRET not in str(e.value)


def test_store_static_session_token_reaches_pyarrow():
    s = Store("s3://b/x?region=us-east-1",
              s3=S3AccessConfig(access_key_id=KEY, secret_access_key=SECRET, session_token=TOKEN))
    assert _fs_kwargs(s.fs)["session_token"] == TOKEN


def test_store_default_unchanged(tmp_path):
    s = Store(str(tmp_path))
    assert s.s3 is None and isinstance(s.fs, pafs.LocalFileSystem)


# ---- native (rust_fetch) --------------------------------------------------

def test_auth_kwargs_mapping_is_exact():
    assert nio._auth_kwargs(None) == {}
    assert nio._auth_kwargs(S3AccessConfig(anonymous=True)) == {"anonymous": True}
    assert nio._auth_kwargs(S3AccessConfig(
        access_key_id=KEY, secret_access_key=SECRET, session_token=TOKEN)) == {
        "access_key_id": KEY, "secret_access_key": SECRET, "session_token": TOKEN}
    assert nio._auth_kwargs(S3AccessConfig(
        access_key_id=KEY, secret_access_key=SECRET))["session_token"] is None


def test_old_rust_fetch_build_never_signs_with_env(monkeypatch):
    """An old rust_fetch without auth kwargs must make the store fall back to
    pyarrow, never construct an env-signed client for an s3: location."""
    import sys
    import types

    built = []

    class OldClient:
        def __init__(self, region=None, concurrency=64, part_size=16 << 20):
            built.append(region)

    monkeypatch.setitem(sys.modules, "rust_fetch", types.SimpleNamespace(Client=OldClient))
    monkeypatch.setitem(nio._NATIVE, "clients", {})
    monkeypatch.setitem(nio._NATIVE, "warned", set())
    for auth in (S3AccessConfig(anonymous=True),
                 S3AccessConfig(access_key_id=KEY, secret_access_key=SECRET)):
        assert nio._native_client("us-east-1", auth) is None
    assert built == []
    assert nio._native_client("us-east-1", None) is not None  # default path unchanged
    assert built == ["us-east-1"]


def test_store_native_reads_use_the_store_auth(monkeypatch):
    """Every native entry point on a Store asks for a client with ITS auth."""
    auth = S3AccessConfig(anonymous=True)
    s = Store("s3://b/x/?region=us-east-1", s3=auth)
    seen = []

    class Spy:
        def head(self, url):
            raise FileNotFoundError(url)

        def dense_plan(self, *a, **k):
            raise FileNotFoundError("x")

    def fake_client(region, s3=None):
        seen.append((region, s3))
        return Spy()

    monkeypatch.setattr(nio, "_native_client", fake_client)
    monkeypatch.setattr(nio, "rust_fetch_dense_enabled", lambda: True)
    monkeypatch.setattr(nio, "rust_fetch_columns_enabled", lambda: True)
    pytest.importorskip("rust_fetch")
    s.read_dense_versioned("b/x/f.parquet", "dense_embedding")
    s.read_columns_fetched("b/x/f.parquet", ["id"])
    s.same_version("b/x/f.parquet", "etag")
    assert len(seen) == 3 and all(r == ("us-east-1", auth) for r in seen), seen

def test_native_clients_keyed_by_auth():
    rf = pytest.importorskip("rust_fetch")
    try:
        rf.Client(anonymous=True)
    except TypeError:
        pytest.skip("installed rust_fetch predates per-location auth; rebuild rust_io")
    anon = nio._native_client("us-east-1", S3AccessConfig(anonymous=True))
    static = nio._native_client("us-east-1", S3AccessConfig(access_key_id=KEY, secret_access_key=SECRET))
    default = nio._native_client("us-east-1", None)
    assert len({id(anon), id(static), id(default)}) == 3
    assert "auth=anonymous" in repr(anon)
    assert "auth=static" in repr(static) and SECRET not in repr(static) and KEY not in repr(static)
    assert "auth=env" in repr(default)
    assert nio._native_client("us-east-1", S3AccessConfig(anonymous=True)) is anon


@pytest.mark.parametrize("kw", [
    {"anonymous": True, "access_key_id": KEY, "secret_access_key": SECRET},
    {"access_key_id": KEY},
    {"secret_access_key": SECRET},
    {"session_token": TOKEN},
    {"access_key_id": "", "secret_access_key": SECRET},
    {"access_key_id": KEY, "secret_access_key": SECRET, "session_token": ""},
])
def test_rust_client_rejects_bad_auth(kw):
    rf = pytest.importorskip("rust_fetch")
    try:
        rf.Client(anonymous=True)
    except TypeError:
        pytest.skip("installed rust_fetch predates per-location auth; rebuild rust_io")
    with pytest.raises(ValueError):
        rf.Client(**kw)


@pytest.mark.skipif(os.environ.get("NOVA_BF_LIVE_S3") != "1",
                    reason="live S3; set NOVA_BF_LIVE_S3=1")
def test_live_public_bucket_anonymous(monkeypatch):
    """Anonymous reads must ignore (here: poisoned) env credentials, on both
    the pyarrow and native paths, and the two must agree byte for byte.
    Downloads one ~141 MB file's dense column."""
    import numpy as np

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAPOISONED00000000")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "poison")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "poison")
    s = Store("s3://fineweb-gte-with-payloads/resharded/00/?region=us-east-1",
              s3=S3AccessConfig(anonymous=True))
    f = "fineweb-gte-with-payloads/resharded/00/train-part0__0200.parquet"
    assert any(p.read_path == f for p in s.list_parquets())
    native = s.read_dense_versioned(f, "dense_embedding")
    assert native is not None, "native read fell back to pyarrow"
    ref = nio.dense_to_2d(s.read_columns(f, ["dense_embedding"]).column(0))
    assert np.array_equal(native[0], ref)


# ---- listing under a prefix-scoped grant ---------------------------------

class _HeadDenied:
    """Local fs whose single-path HEAD is denied (as S3 does for `prefix`
    when only `prefix/*` is granted) but whose listing works."""

    def __init__(self, inner):
        self.inner = inner

    def get_file_info(self, what):
        if isinstance(what, str):
            raise OSError("AWS Error ACCESS_DENIED during HeadObject operation")
        return self.inner.get_file_info(what)


def _s3_like_store(root):
    s = Store(str(root))
    s.is_s3 = True
    s.fs = _HeadDenied(pafs.LocalFileSystem())
    s._loader_key = lambda p: p
    return s


def test_list_falls_back_to_listing_when_head_denied(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    (tmp_path / "00").mkdir()
    pq.write_table(pa.table({"x": [1]}), tmp_path / "00" / "a.parquet")
    files = _s3_like_store(tmp_path).list_parquets()
    assert [f.read_path.rsplit("/", 2)[-2:] for f in files] == [["00", "a.parquet"]]


def test_list_reraises_head_error_when_listing_is_empty(tmp_path):
    with pytest.raises(OSError, match="ACCESS_DENIED"):
        _s3_like_store(tmp_path).list_parquets()


@pytest.mark.skipif(os.environ.get("NOVA_BF_LIVE_S3") != "1",
                    reason="live S3; set NOVA_BF_LIVE_S3=1")
def test_live_list_prefix_scoped_root():
    s = Store("s3://fineweb-gte-with-payloads/resharded/?region=us-east-1",
              s3=S3AccessConfig(anonymous=True))
    assert len(s.list_parquets()) > 1000


def test_bad_tuning_env_does_not_deadlock(monkeypatch):
    """_env_int warns via _warn_once, which takes _NATIVE_LOCK; it must not
    run while _native_client holds that lock (it used to deadlock)."""
    import threading
    pytest.importorskip("rust_fetch")
    monkeypatch.setenv("NOVA_BF_RUST_FETCH_CONC", "not-a-number")
    monkeypatch.setitem(nio._NATIVE, "warned", set())
    t = threading.Thread(target=nio._native_client, args=("us-east-1", None), daemon=True)
    t.start()
    t.join(20)
    assert not t.is_alive(), "_native_client deadlocked on a bad tuning env var"


# ---- round-2 review regressions ---------------------------------------------

def test_yaml_syntax_error_never_echoes_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("T_SECRET", SECRET)
    p = tmp_path / "c.yaml"
    p.write_text("corpus:\n  path: s3://b/c/\n  s3:\n    access_key_id: AKIAX\n"
                 "     secret_access_key: ${T_SECRET}\n")      # mis-indented
    with pytest.raises(ValueError, match=r"invalid YAML near line 5") as e:
        load_config(str(p))
    assert SECRET not in str(e.value) and SECRET not in repr(e.value)
    assert e.value.__cause__ is None and e.value.__suppress_context__


@pytest.mark.parametrize("path, match", [
    ("/tmp/out", "not an s3:// path"),
    (f"s3://{KEY}:{SECRET}@b/out", "embedded"),
])
def test_bad_s3_location_fails_at_config_load(path, match):
    from nova_bf.config import OutputConfig
    with pytest.raises(ValidationError, match=match) as e:
        OutputConfig(path=path, s3=S3AccessConfig(anonymous=True))
    assert SECRET not in str(e.value)


def test_masked_secret_from_a_dump_is_rejected():
    with pytest.raises(ValidationError, match="masked placeholder"):
        S3AccessConfig(access_key_id="**********", secret_access_key="**********")


@pytest.mark.parametrize("field", ["access_key_id", "secret_access_key", "session_token"])
def test_explicit_null_credential_field_is_rejected(field):
    kw = {"access_key_id": KEY, "secret_access_key": SECRET, "session_token": TOKEN}
    kw[field] = None                      # what `field: ${UNSET:-}` (unquoted) parses to
    with pytest.raises(ValidationError, match="no value|given together"):
        S3AccessConfig(**kw)


def test_list_reraises_head_error_when_listing_also_fails():
    """Real S3 raises (not []) when listing a denied/missing prefix."""
    class BothDenied:
        def get_file_info(self, what):
            if isinstance(what, str):
                raise OSError("AWS Error ACCESS_DENIED during HeadObject operation")
            raise OSError("AWS Error ACCESS_DENIED during ListObjectsV2 operation")

    s = Store("s3://b/x?region=us-east-1", s3=S3AccessConfig(anonymous=True))
    s.fs = BothDenied()
    with pytest.raises(OSError, match="HeadObject"):
        s.list_parquets()


def test_local_head_error_is_not_retried_as_a_listing(tmp_path):
    s = Store(str(tmp_path))
    calls = []

    class Fs:
        def get_file_info(self, what):
            calls.append(what)
            raise OSError("local failure")

    s.fs = Fs()
    with pytest.raises(OSError, match="local failure"):
        s.list_parquets()
    assert len(calls) == 1


def test_every_store_in_compute_and_merge_gets_its_sections_auth(tmp_path, monkeypatch):
    """End to end: compute (2 ranks, threshold exchange on) and merge build
    every Store for a section with THAT section's s3 block -- never None."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from nova_bf.compute import run_compute
    from nova_bf.config import (BruteForceConfig, CorpusConfig, OutputConfig,
                                QueriesConfig, SearchSpec)
    from nova_bf.merge import run_merge

    monkeypatch.setenv("NOVA_BF_DEVICE", os.environ.get("NOVA_BF_TEST_DEVICE", "cpu"))
    monkeypatch.setenv("NOVA_BF_RUST_FETCH_DENSE", "0")    # fake s3: no real network
    monkeypatch.setenv("NOVA_BF_RUST_FETCH_COLUMNS", "0")
    root = tmp_path / "fake_s3"
    rng = np.random.default_rng(0)
    (root / "b" / "corpus").mkdir(parents=True)
    (root / "b" / "out").mkdir(parents=True)
    for i in range(4):
        v = rng.standard_normal((12, 8)).astype(np.float32)
        pq.write_table(pa.table({"dense_embedding": pa.array(v.tolist(), pa.list_(pa.float32()))}),
                       root / "b" / "corpus" / f"part{i}.parquet")
    q = rng.standard_normal((5, 8)).astype(np.float32)
    pq.write_table(pa.table({"dense_embedding": pa.array(q.tolist(), pa.list_(pa.float32()))}),
                   root / "b" / "q.parquet")

    seen = []
    real = nio._fs_and_path

    fsspec = pytest.importorskip("fsspec")
    # Local stand-in for S3: like S3 (and unlike pyarrow's LocalFileSystem),
    # writing a key needs no parent "directory" to exist first.
    local = pafs.PyFileSystem(pafs.FSSpecHandler(fsspec.filesystem("file", auto_mkdir=True)))

    def fake_fs_and_path(uri, s3=None):
        seen.append((uri, s3))
        if uri.startswith("s3://"):
            return pafs.SubTreeFileSystem(str(root), local), uri[len("s3://"):]
        return real(uri, s3)

    monkeypatch.setattr(nio, "_fs_and_path", fake_fs_and_path)
    auth = {
        "s3://b/corpus/": S3AccessConfig(anonymous=True),
        "s3://b/q.parquet": S3AccessConfig(access_key_id="AKIAQUERYKEY00000000", secret_access_key="q-secret"),
        "s3://b/out/": S3AccessConfig(access_key_id="AKIAOUTPUTKEY0000000", secret_access_key="o-secret"),
    }
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path="s3://b/corpus/", s3=auth["s3://b/corpus/"]),
        queries=QueriesConfig(path="s3://b/q.parquet", s3=auth["s3://b/q.parquet"]),
        output=OutputConfig(path="s3://b/out/", s3=auth["s3://b/out/"]),
        params={"threshold_exchange": True},
        searches=[SearchSpec(name="d", vector_type="dense", metric="dot", k=3)])
    run_compute(cfg, num_jobs=2, job_rank=0)
    run_compute(cfg, num_jobs=2, job_rank=1)
    out = run_merge(cfg)
    assert list(out) == ["d"]
    assert (root / out["d"]).exists()       # merged result written via the output auth
    by_loc = {}
    for uri, s3 in seen:
        loc = next(k for k in auth if uri.startswith(k.rstrip("/")))
        by_loc.setdefault(loc, []).append(s3)
    assert set(by_loc) == set(auth), by_loc.keys()
    for loc, got in by_loc.items():
        assert all(g == auth[loc] for g in got), (loc, got)


# ---- round-3 review regressions ---------------------------------------------

@pytest.mark.parametrize("section", ["corpus", "queries", "output"])
@pytest.mark.parametrize("scheme", ["s3", "https"])
def test_uri_credentials_rejected_without_a_block_and_never_echoed(section, scheme):
    from nova_bf.config import CorpusConfig, OutputConfig, QueriesConfig
    cls = {"corpus": CorpusConfig, "queries": QueriesConfig, "output": OutputConfig}[section]
    with pytest.raises(ValidationError, match="embedded") as e:
        cls(path=f"{scheme}://{KEY}:{SECRET}@b/x")
    assert KEY not in str(e.value) and SECRET not in str(e.value)


@pytest.mark.parametrize("value", ["*SECRETALIAS99", "!SECRETTAG99", "!SECRETHANDLE!x"])
def test_yaml_alias_or_tag_errors_never_echo_secrets(tmp_path, monkeypatch, value):
    monkeypatch.setenv("T_SECRET", value)
    p = tmp_path / "c.yaml"
    p.write_text("corpus:\n  path: s3://b/c/\n  s3:\n    access_key_id: AKIAX\n"
                 "    secret_access_key: ${T_SECRET}\n")
    with pytest.raises(ValueError, match="invalid YAML") as e:
        load_config(str(p))
    assert value[1:] not in str(e.value) and "SECRET" not in str(e.value)


@pytest.mark.parametrize("cls_name, kw", [
    ("S3AccessConfig", {"anonymous": True, "access_key_id": KEY, "secret_access_key": SECRET}),
    ("CorpusConfig", {"path": "/local", "s3": {"access_key_id": KEY, "secret_access_key": SECRET}}),
    ("QueriesConfig", {"path": "/local", "s3": {"access_key_id": KEY, "secret_access_key": SECRET}}),
    ("OutputConfig", {"path": "/local", "s3": {"access_key_id": KEY, "secret_access_key": SECRET}}),
    ("OutputConfig", {"path": "s3://b/o", "s3": {"access_key_id": KEY, "secret_acess_key": SECRET}}),
])
def test_direct_construction_errors_never_echo_secrets(cls_name, kw):
    """Built in code (not via load_config/the root model), each section model
    must hide its own input too."""
    import nova_bf.config as C
    with pytest.raises(ValidationError) as e:
        getattr(C, cls_name)(**kw)
    assert KEY not in str(e.value) and SECRET not in str(e.value), str(e.value)


def test_native_warnings_redact_credentials(monkeypatch):
    monkeypatch.setenv("AWS_SESSION_TOKEN", "ENVTOKENVALUE")
    s3 = S3AccessConfig(access_key_id=KEY, secret_access_key=SECRET, session_token=TOKEN)
    body = f"403 <Error><Token>{TOKEN}</Token><Key>{KEY}</Key><S>{SECRET}</S><E>ENVTOKENVALUE</E></Error>"
    out = nio._redact(OSError(body), s3)
    for v in (KEY, SECRET, TOKEN, "ENVTOKENVALUE"):
        assert v not in out
    assert "403" in out and nio._redact("plain", None) == "plain"


def test_cli_merge_fanout_store_uses_the_output_auth(tmp_path, monkeypatch):
    """The parent's legacy-manifest Store (cli._merge_fanout) carries output.s3."""
    import types
    import nova_bf.cli as cli_mod
    import nova_bf.merge as merge_mod

    monkeypatch.setenv("T_KEY", KEY)
    monkeypatch.setenv("T_SECRET", SECRET)
    p = tmp_path / "c.yaml"
    p.write_text("corpus:\n  path: /tmp/c/\nqueries:\n  path: /tmp/q.parquet\n"
                 "output:\n  path: s3://b/out/\n  s3:\n    access_key_id: ${T_KEY}\n"
                 "    secret_access_key: ${T_SECRET}\n"
                 "searches:\n  - {name: a, k: 10, metric: dot}\n")
    got = []
    monkeypatch.setattr(merge_mod, "preflight_searches", lambda cfg: None)
    monkeypatch.setattr(merge_mod, "drop_legacy_manifest",
                        lambda store, cfg, names: got.append(store.s3))
    monkeypatch.setattr(nio, "_fs_and_path", lambda uri, s3=None: (pafs.LocalFileSystem(), str(tmp_path)))

    class Done:
        pid, returncode = 4242, 0
        def poll(self): return 0
        def wait(self, timeout=None): return 0

    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: Done())
    assert cli_mod._merge_fanout(str(p), ["a"], 1) == 0
    assert len(got) == 1 and got[0] is not None
    assert got[0].secret_access_key.get_secret_value() == SECRET


@pytest.mark.parametrize("path", [
    "s3://AKIAEXAMPLE:wJalrXUtnFEMI/K7MDENG/bPxRfiCYSECRET@bucket/c/?region=us-east-1",
    "s3://AKIAEXAMPLE:abc?SECRETdef@bucket/c/",
    "s3://AKIAEXAMPLE:abc#SECRETdef@bucket/c/",
])
def test_uri_credentials_with_slash_in_secret_rejected_at_load(path):
    """AWS secret keys often contain `/`, which hides the `@` from urlsplit."""
    from nova_bf.config import CorpusConfig
    for s3 in (None, S3AccessConfig(anonymous=True)):
        with pytest.raises(ValidationError, match="embedded") as e:
            CorpusConfig(path=path, s3=s3)
        assert "SECRET" not in str(e.value) and "AKIAEXAMPLE" not in str(e.value)


@pytest.mark.parametrize("path", [
    "s3://bucket/key@with-at/x.parquet", "s3://bucket/a:b/c@d", "/local/a@b:c",
    "file:///x@y/z", "s3://bucket/k=v/part.parquet", "s3://bucket?region=us-east-1",
])
def test_valid_paths_with_at_or_colon_still_accepted(path):
    from nova_bf.config import CorpusConfig
    assert CorpusConfig(path=path).path == path
