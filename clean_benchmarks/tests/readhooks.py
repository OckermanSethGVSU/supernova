"""Picklable ``read_fn`` stand-ins for ``ParquetSource(read_fn=...)``.

Reader PROCESSES cannot see a monkeypatch made in the test process, so the
tests that need a slow / failing / crashing read pass one of these (or a
``functools.partial`` of one) instead; spawned readers import this module by
name (the tests directory is on ``sys.path``, which spawn hands down). They
work the same with ``reader="thread"``.
"""

import os
import time

import pyarrow.fs as pafs

from bench.data import sources
from bench.timing import now


def slow_read(fs, path, *a, delays=None, default=0.0, **kw):
    """``read_file`` after sleeping ``delays[basename]`` (else ``default``)
    seconds; ``rec["w_start"]`` is moved back to cover the sleep, so the
    worker interval [w_start, w_end] is the whole read (for in-flight
    counting across processes: perf_counter is CLOCK_MONOTONIC, one clock
    for every process)."""
    t = now()
    time.sleep((delays or {}).get(os.path.basename(path), default))
    x = sources.read_file(fs, path, *a, **kw)
    rec = kw.get("rec")
    if rec is not None:
        rec["w_start"] = t
        rec["args"] = dict(mode=a[4], range_concurrency=a[5], range_bytes=a[6],
                           pid=os.getpid(), pa_cpu=sources.pa.cpu_count(),
                           pa_io=sources.pa.io_thread_count())
    return x


def failing_read(fs, path, *a, fail=None, **kw):
    """``OSError`` for the file whose basename is ``fail`` (every file when
    None), a normal read otherwise."""
    if fail is None or os.path.basename(path) == fail:
        raise OSError("s3 went away")
    return sources.read_file(fs, path, *a, **kw)


def crashing_read(fs, path, *a, crash=None, **kw):
    """The reader process dies (``os._exit``, no cleanup, like an OOM kill)
    on file ``crash``."""
    if crash is None or os.path.basename(path) == crash:
        os._exit(3)
    return sources.read_file(fs, path, *a, **kw)


def publish_then_crash(fs, path, *a, crash=None, **kw):
    """On file ``crash``: decode, write the block into the ring slot the
    parent named for this task (``sources._TASK_SLOT``), then die before
    replying: a slot the parent never hears back about."""
    if crash is None or os.path.basename(path) == crash:
        x = sources.read_file(fs, path, *a, **kw)
        sources.publish_to_slot(x, sources._TASK_SLOT, path)
        os._exit(3)
    return sources.read_file(fs, path, *a, **kw)


def module_probe(fs, path, *a, **kw):
    """``read_file``, plus which heavy modules this (reader) process has
    imported and the script it re-ran as its main, if any, in
    ``rec["mods"]`` (``__mp_main__`` itself is always there: multiprocessing
    aliases it to ``__main__``)."""
    import sys
    x = sources.read_file(fs, path, *a, **kw)
    rec = kw.get("rec")
    if rec is not None:
        main = getattr(sys.modules.get("__main__"), "__file__", None)
        rec["mods"] = sorted(m for m in ("faiss", "torch", "bench.systems.harness_cli") if m in sys.modules) + \
            ([f"main={os.path.basename(main)}"] if main else [])
    return x


def warm_probe(fs, path, *a, **kw):
    """``read_file``, plus whether ``pyarrow.dataset`` (which
    ``pq.read_table`` imports lazily, ~0.1 s) was already imported when this
    read began, in ``rec["dataset_imported"]``."""
    import sys
    before = "pyarrow.dataset" in sys.modules
    x = sources.read_file(fs, path, *a, **kw)
    rec = kw.get("rec")
    if rec is not None:
        rec["dataset_imported"] = before
        rec["pid"] = os.getpid()
    return x


def combine_spy(fs, path, *a, **kw):
    """``read_file``, plus how many times it called
    ``sources._combine_chunks`` (``rec["combines"]``) and whether the block it
    returned is an array over the ``out`` buffer it was handed
    (``rec["in_out"]``; False without one)."""
    calls = []
    real = sources._combine_chunks

    def spy(chunked):
        calls.append(chunked.num_chunks)
        return real(chunked)

    sources._combine_chunks = spy
    try:
        x = sources.read_file(fs, path, *a, **kw)
    finally:
        sources._combine_chunks = real
    rec = kw.get("rec")
    if rec is not None:
        out = kw.get("out")
        rec["combines"] = len(calls)
        rec["in_out"] = out is not None and x.base is out
    return x


def write_version(path, version, vec_type=None, D=8, N=64):
    """Write ``path`` as version ``version``: N rows of column ``s`` ("a" for
    version 1, "b" otherwise) and a D-wide fixed_size_list ``embedding``
    filled with float(version) (stored as ``vec_type``, default float32)."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    x = np.full((N, D), float(version), np.float32)
    s = ["a"] * N if version == 1 else ["b"] * N
    t = pa.table({"s": s, "embedding": pa.FixedSizeListArray.from_arrays(
        pa.array(x.reshape(-1)).cast(vec_type or pa.float32()), D)})
    pq.write_table(t, path)


def changes_after_dense_read(fs, path, *a, rewrite=None, **kw):
    """``read_file`` with the object rewritten (``rewrite()``) right after the
    native dense read returns, before the filter columns are fetched: the
    object changes between the two reads of one file."""
    real = sources.read_native

    def changing(*x, **y):
        out = real(*x, **y)
        rewrite()
        return out

    sources.read_native = changing
    try:
        return sources.read_file(fs, path, *a, **kw)
    finally:
        sources.read_native = real


class LoggedLocalFS(pafs.FileSystemHandler):
    """A ``pyarrow.fs.PyFileSystem`` handler over the local filesystem that
    appends ``"<pid> <op> <path>"`` to the file ``log`` on every metadata
    access / open, from whichever process uses it (the pickled ``fs`` a reader
    process gets included): which process touched its filesystem, and when."""

    def __init__(self, log):
        self.log, self.fs = log, pafs.LocalFileSystem()

    def _note(self, op, path):
        with open(self.log, "a") as fh:
            fh.write(f"{os.getpid()} {op} {path}\n")

    def __eq__(self, other):
        return isinstance(other, LoggedLocalFS) and other.log == self.log

    def equals(self, other):
        return self == other

    def get_type_name(self):
        return "logged-local"

    def normalize_path(self, path):
        return self.fs.normalize_path(path)

    def get_file_info(self, paths):
        self._note("info", paths)
        return self.fs.get_file_info(paths)

    def get_file_info_selector(self, selector):
        return self.fs.get_file_info(selector)

    def create_dir(self, path, recursive):
        self.fs.create_dir(path, recursive=recursive)

    def delete_dir(self, path):
        self.fs.delete_dir(path)

    def delete_dir_contents(self, path, missing_dir_ok=False):
        self.fs.delete_dir_contents(path, missing_dir_ok=missing_dir_ok)

    def delete_root_dir_contents(self):
        raise NotImplementedError

    def delete_file(self, path):
        self.fs.delete_file(path)

    def move(self, src, dest):
        self.fs.move(src, dest)

    def copy_file(self, src, dest):
        self.fs.copy_file(src, dest)

    def open_input_stream(self, path):
        self._note("open", path)
        return self.fs.open_input_stream(path)

    def open_input_file(self, path):
        self._note("open", path)
        return self.fs.open_input_file(path)

    def open_output_stream(self, path, metadata):
        return self.fs.open_output_stream(path, metadata=metadata)

    def open_append_stream(self, path, metadata):
        return self.fs.open_append_stream(path, metadata=metadata)


class WarmUpPoolProbe:
    """A ``read_fn`` (plain ``read_file``) whose UNPICKLING, i.e. in the
    spawned reader process, before ``_reader_main`` runs, wraps that
    process's ``sources._warm_up_reads`` so that the warm-up read appends
    ``"<pid> <pa.cpu_count()> <pa.io_thread_count()>"`` to the file ``log``
    from inside it: the pools the warm-up read actually ran with, which are
    the assigned ones only if the reader configures its pools first. Pickling
    (the parent) and thread readers (never pickled) leave ``sources`` alone."""

    def __init__(self, log):
        self.log = log

    def __setstate__(self, state):
        self.__dict__.update(state)
        real, log = sources._warm_up_reads, self.log

        def probed():
            with open(log, "a") as fh:
                fh.write(f"{os.getpid()} {sources.pa.cpu_count()} "
                         f"{sources.pa.io_thread_count()}\n")
            return real()

        sources._warm_up_reads = probed

    def __call__(self, fs, path, *a, **kw):
        return sources.read_file(fs, path, *a, **kw)

