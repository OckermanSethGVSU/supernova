"""Load corpus blocks for benchmark engines.

ParquetSource streams local or S3 files through an ordered pool of reader
processes using shared memory. InMemorySource holds decoded data in RAM
for verification.

Timed reads use nova-bf's native Rust reader, while verification uses
independent PyArrow decoding. Blocks preserve their stored dtype unless
conversion is requested.

Source statistics track I/O and decoding; the caller separately measures
time spent waiting for blocks.
"""

import contextlib
import errno
import fcntl
import functools
import glob
import mmap
import multiprocessing as mp
import multiprocessing.connection as mpc
import os
import resource
import secrets
import signal
import struct
import sys
import threading
import traceback
import types
import warnings
import weakref
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import _RemoteTraceback
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from bench.timing import Stages, now

# the timed read mode (--read-mode); read_file also takes ARROW (untimed pyarrow
# decode for the verify reload and the query file)
READ_MODES = ("native",)
ARROW = "arrow"
FILE_MODES = (ARROW, "native")
# nova-bf's rust_fetch defaults (io.py): in-flight GETs per process-wide client
# (NOVA_BF_RUST_FETCH_CONC), bytes per GET (..._PART_MIB), row groups in flight per
# file (..._CHUNKS)
NATIVE_CONCURRENCY = 128
NATIVE_PART_BYTES = 16 << 20
NATIVE_CHUNKS_IN_FLIGHT = 8
# nova-bf reads with pyarrow when any of these is set (custom S3 endpoint);
# native mode refuses instead (it never silently reads another way)
NATIVE_ENDPOINT_ENV = ("AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT")
RANGE_SCOPES = ("total", "per-file")
# float types carried to the device at their stored width (they widen to
# float32 exactly); anything else is widened to float32 on the host
STORED_DTYPES = (np.dtype(np.float16), np.dtype(np.float32))


@dataclass
class Block:
    # global row id of the file's first row; also vectors[0]'s, except in a
    # reader-compacted block (``compact`` set), whose row i is file row keep[i]
    offset: int
    vectors: np.ndarray  # (n, d), C-contiguous
    path: str = ""
    info: dict | None = None  # this read's stamps (read_file ``rec``), if timed
    filter: object = None     # filters.reader.FilterBlock when the source filters
    # filters.reader.Compacted when the reader compacted the block (--filter-mode
    # gather / --filter-compact union): ``vectors`` are then only the kept
    # rows (file rows ``compact.keep``), ``filter`` is over them (union), and
    # the file's own row count is ``compact.n_rows``
    compact: object = None


@dataclass
class FileRef:
    path: str
    size: int


@dataclass
class Footer:
    """What a file's parquet footer says about the selected column."""
    num_rows: int
    num_row_groups: int
    column_bytes: int    # sum of the column's total_compressed_size over row groups
    dim: int | None      # fixed_size_list: its list_size; list / large_list: leaf
                         # values / rows of the first non-empty row group
    value_type: str      # list value type as Arrow prints it (halffloat, float, ...)
    value_bits: int | None = None  # its bit width if a float type (16 / 32 / 64), else None
    max_rg_rows: int | None = None  # rows of the file's largest row group
    list_kind: str | None = None    # list / large_list / fixed_size_list (else None)
    compression: str | None = None  # the column's codec (first row group), e.g. SNAPPY
    # sparse struct<indices: list<int>, values: list<float>> columns
    # (sparse.SparseCodec; None otherwise): an upper bound on the file's
    # nonzeros (the indices leaf's value count over the row groups, which
    # also counts each empty list's placeholder) and the values' float width
    sparse_nnz: int | None = None
    sparse_value_bits: int | None = None
    # multi-vector list<list<float>> columns (multivector.MultiVectorCodec; None
    # otherwise): an upper bound on the file's token values (n_tok * dim: the
    # leaf's value count over the row groups, which also counts each null or
    # empty list's placeholder) and the values' float width
    mv_values: int | None = None
    mv_value_bits: int | None = None


def _cgroup_cpu_quota():
    """CPU count allowed by the cgroup quota, or None if unlimited (copied
    from nova-bf compute.py). v2 then v1; failed probes are ignored."""
    try:  # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota, period = fh.read().split()[:2]
        if quota != "max":
            q, per = int(quota), int(period)
            if q > 0 and per > 0:
                return max(1, -(-q // per))  # ceil: a 1.5-cpu quota allows 2
    except Exception:  # noqa: BLE001 - absent, unreadable, or an odd layout
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            q = int(fh.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            per = int(fh.read().strip())
        if q > 0 and per > 0:
            return max(1, -(-q // per))
    except Exception:  # noqa: BLE001
        pass
    return None


def usable_cpu_count():
    """CPUs this process can actually use: ``os.cpu_count()`` capped by CPU
    affinity and the cgroup quota (nova-bf ``_usable_cpu_count``)."""
    n = os.cpu_count() or 1
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):  # not Linux, or not permitted
        pass
    quota = _cgroup_cpu_quota()
    if quota:
        n = min(n, quota)
    return max(1, n)


def default_workers():
    """Default reader count: all usable CPUs but one."""
    return max(1, usable_cpu_count() - 1)


def configure_arrow_pools(cpu_thread_count=None, io_thread_count=None):
    """Size pyarrow's decode (cpu) and IO pools; returns the resulting
    ``(pa.cpu_count(), pa.io_thread_count())``. Must run before the first
    parquet read (as nova-bf's ``configure_thread_pools``).

    pyarrow takes its cpu pool size from OMP_NUM_THREADS, and SkyPilot exports
    OMP_NUM_THREADS=1: a one-thread decode pool throttles every reader at once
    and looks just like an I/O bottleneck. So the cpu pool is always set
    explicitly: ``cpu_thread_count`` or, when unset/0, ``usable_cpu_count()``.
    The io pool is only set when ``io_thread_count`` > 0 (else Arrow's default)."""
    if io_thread_count and io_thread_count > 0:
        pa.set_io_thread_count(io_thread_count)
    if not cpu_thread_count or cpu_thread_count <= 0:
        cpu_thread_count = usable_cpu_count()
    pa.set_cpu_count(cpu_thread_count)
    return pa.cpu_count(), pa.io_thread_count()


def is_vector_list(typ):
    """True for the column types a block is decoded from: ``list``,
    ``large_list`` and ``fixed_size_list`` (of a float type; anything else
    fails in ``table_vectors``). A ``fixed_size_list`` carries d in its type 
    and has no offsets buffer; the others carry d per row."""
    return (pa.types.is_list(typ) or pa.types.is_large_list(typ)
            or pa.types.is_fixed_size_list(typ))


def read_footer(fs, path, column):
    """Footer-only facts about ``column`` (one small read: ``read_metadata``).

    A 0-row file raises ValueError ("empty file"), a missing column KeyError.
    d is a ``fixed_size_list``'s ``list_size`` (from the Arrow schema pyarrow
    stores in the footer), else leaf values / rows of the first non-empty
    row group."""
    return footer_from_metadata(pq.read_metadata(path, filesystem=fs), path, column)


def footer_from_metadata(md, path, column):
    """``read_footer`` from an already-read ``FileMetaData`` ``md``."""
    if md.num_rows == 0:
        raise ValueError(f"{path}: empty file")
    col_bytes, dim, leaves, codec = 0, None, set(), None
    schema = md.schema.to_arrow_schema()
    sparse = _sparse_struct(schema.field(column).type) \
        if schema.get_field_index(column) >= 0 else None
    sparse_nnz = 0 if sparse else None
    mv = _mv_value_type(schema.field(column).type) \
        if schema.get_field_index(column) >= 0 else None
    mv_values = 0 if mv is not None else None
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        for j in range(rg.num_columns):
            cc = rg.column(j)
            if cc.path_in_schema.split(".")[0] != column:
                continue
            leaves.add(cc.path_in_schema)
            col_bytes += cc.total_compressed_size
            codec = codec or cc.compression
            if sparse and cc.path_in_schema.split(".")[1:2] == ["indices"]:
                sparse_nnz += cc.num_values
            if mv is not None:
                mv_values += cc.num_values  # the one leaf: every token value
            # list<float> with no nulls: leaf values = rows * d
            if dim is None and rg.num_rows and cc.num_values % rg.num_rows == 0:
                dim = cc.num_values // rg.num_rows
    if not leaves:
        raise KeyError(f"{path}: no column {column!r}")
    if len(leaves) > 1:
        dim = None  # a struct, not list<float>: let decode complain
    typ = schema.field(column).type
    if pa.types.is_fixed_size_list(typ):
        dim = typ.list_size
    vtyp = typ.value_type if hasattr(typ, "value_type") else typ
    bits = vtyp.bit_width if pa.types.is_floating(vtyp) else None
    kind = next((k for k, f in LIST_KINDS if f(typ)), None)
    rg_rows = max(md.row_group(i).num_rows for i in range(md.num_row_groups))
    sparse_bits = sparse[1].value_type.bit_width if sparse else None
    if mv is not None:
        dim = None  # per token, not per row: the decode finds it
    return Footer(md.num_rows, md.num_row_groups, col_bytes, dim, str(vtyp), bits,
                  rg_rows, kind, codec, sparse_nnz, sparse_bits, mv_values,
                  mv.bit_width if mv is not None else None)


def _mv_value_type(typ):
    """Float value type of a multi-vector ``list<list<float>>`` type, else None.

    Same as multivector.mv_value_type, repeated here to avoid importing it."""
    lists = (pa.types.is_list, pa.types.is_large_list)
    if not any(f(typ) for f in lists):
        return None
    inner = typ.value_type
    if not (any(f(inner) for f in lists) or pa.types.is_fixed_size_list(inner)):
        return None
    vt = inner.value_type
    return vt if pa.types.is_floating(vt) else None


def _sparse_struct(typ):
    """``(indices list type, values list type)`` of a sparse struct type, else None.

    Same as sparse.sparse_fields, repeated here to avoid importing it."""
    if not pa.types.is_struct(typ):
        return None
    names = {typ.field(i).name for i in range(typ.num_fields)}
    if not {"indices", "values"} <= names:
        return None
    it, vt = typ.field("indices").type, typ.field("values").type
    if not all(pa.types.is_list(t) or pa.types.is_large_list(t) for t in (it, vt)):
        return None
    if not (pa.types.is_integer(it.value_type) and pa.types.is_floating(vt.value_type)):
        return None
    return it, vt


LIST_KINDS = (("fixed_size_list", pa.types.is_fixed_size_list),
              ("large_list", pa.types.is_large_list), ("list", pa.types.is_list))


def choose_read_mode(mode, footer, size):
    """``(mode, column_share)``: the mode, checked, and the column's share of the
    file's bytes (column bytes / file bytes, from its footer)."""
    if mode not in FILE_MODES:
        raise ValueError(f"read mode {mode!r} not in {FILE_MODES}")
    share = footer.column_bytes / size if size else 0.0
    return mode, share


def resolve_read_mode(fs, ref, column, mode):
    """``choose_read_mode`` on ``ref``'s footer (one tiny read)."""
    if ref.size == 0:
        raise ValueError(f"{ref.path}: empty file")
    return choose_read_mode(mode, read_footer(fs, ref.path, column), ref.size)


def range_concurrency_for(readers, total=NATIVE_CONCURRENCY, scope="total"):
    """Per-file range concurrency when ``readers`` files are fetched at once.

    Scope ``total`` (the harness default): ``total`` in-flight GETs shared by
    all readers (at least one each). Scope ``per-file``: every reader gets
    ``total``, which is what nova-bf really does (io.py only suggests dividing;
    compute.py passes 24 per file x io_workers)."""
    if scope not in RANGE_SCOPES:
        raise ValueError(f"range scope {scope!r} not in {RANGE_SCOPES}")
    if scope == "per-file":
        return max(1, total)
    return max(1, total // max(1, readers))


def list_parquet(uri, recursive=False):
    """List the parquet files of ``uri``; returns ``(fs, [FileRef])``.

    ``uri`` is a local path or any URI ``pyarrow.fs.FileSystem.from_uri``
    understands (``s3://bucket/prefix``). A directory/prefix lists its
    ``*.parquet`` files (non-recursive unless ``recursive``), sorted by full
    path so row offsets are deterministic (nova-bf sorts a recursive listing
    the same way); a file is used as-is."""
    if "://" not in uri:
        uri = os.path.abspath(uri)
    fs, path = pafs.FileSystem.from_uri(uri)
    path = path.rstrip("/") or path  # "s3://bucket/prefix/" lists like the prefix
    info = fs.get_file_info(path)
    if info.type == pafs.FileType.NotFound:
        raise FileNotFoundError(uri)
    if info.type == pafs.FileType.File:
        return fs, [FileRef(info.path, info.size)]
    infos = fs.get_file_info(pafs.FileSelector(path, recursive=recursive))
    refs = sorted(
        (FileRef(i.path, i.size) for i in infos
         if i.type == pafs.FileType.File and i.path.endswith(".parquet")),
        key=lambda r: r.path,
    )
    if not refs:
        raise FileNotFoundError(f"no *.parquet files under {uri}")
    return fs, refs


def _combine_chunks(chunked):
    """``chunked.combine_chunks()``, a module function so tests can count calls."""
    return chunked.combine_chunks()


def table_vectors(table, column, out=None):
    """Arrow table -> ``(n, d)`` C-contiguous array in the stored dtype.

    float16 / float32 stay as stored; anything else is widened to float32, as
    nova-bf's ``dense_to_2d``. Zero-copy from the Arrow buffer when possible
    (one chunk, no nulls). The column is a ``list`` / ``large_list`` (every
    row's length checked: ragged is an error) or a ``fixed_size_list`` (d is
    its ``list_size``; no lengths to check, no offsets to read).

    With ``out`` (a writable buffer, e.g. a reader process's ring slot) the block
    is written into the start of ``out`` instead, each chunk (row group)
    copied straight into its row range, and the result is an array over
    ``out``: one copy per block, where ``combine_chunks`` plus a later copy
    into the slot would be two for a multi-row-group file."""
    chunked = table.column(column)
    col = _combine_chunks(chunked) if out is None else chunked
    n = len(col)
    if col.null_count:
        raise ValueError(f"column {column!r} has {col.null_count} null rows")
    if not is_vector_list(col.type):
        raise TypeError(f"column {column!r} is {col.type}, expected list<float>, "
                        "large_list<float> or fixed_size_list<float>")
    if n == 0:
        raise ValueError(f"column {column!r} has no rows")
    if pa.types.is_fixed_size_list(col.type):
        d = col.type.list_size
    else:
        lens = pc.list_value_length(col)
        mm = pc.min_max(lens).as_py()
        if mm["min"] != mm["max"]:
            raise ValueError(f"ragged {column!r}: list lengths {mm['min']}..{mm['max']}")
        d = mm["min"]
    if d == 0:
        # every row an empty list: no vectors to search (the preload's footer
        # would read d = 1 from the empty lists' placeholder leaves)
        raise ValueError(f"column {column!r} has empty lists (d = 0)")
    if out is not None:
        return _chunks_into(chunked, column, n, d, out)
    # flatten() honours a sliced/offset list array (``.values`` would not),
    # fixed_size_list included (its child is sliced at offset x d).
    flat = col.flatten()
    if flat.null_count:
        raise ValueError(f"column {column!r} has null elements")
    x = flat.to_numpy(zero_copy_only=False).reshape(n, d)
    if x.dtype not in STORED_DTYPES:
        x = x.astype(np.float32)
    return np.ascontiguousarray(x)


def block_dtype(list_type):
    """The dtype ``table_vectors`` returns for a ``list_type`` column (list /
    large_list / fixed_size_list; stored float16 / float32, anything else
    float32)."""
    dt = np.dtype(list_type.value_type.to_pandas_dtype())
    return dt if dt in STORED_DTYPES else np.dtype(np.float32)


def _check_slot_room(nbytes, mm, path=""):
    if nbytes > len(mm):
        raise ValueError(f"{path}: decoded block is {nbytes} B, larger than a "
                         f"shared-memory ring slot ({len(mm)} B, sized from the "
                         "footers: rows x d x itemsize)")


def _chunks_into(chunked, column, n, d, out):
    """``table_vectors``' ``out`` path: copy each chunk into its rows of ``out``.

    Chunk by chunk: flatten (zero-copy for a float chunk without nulls) and copy
    into rows ``[row, row + m)`` of an ``(n, d)`` array over ``out`` (numpy copy,
    GIL released; a float64 chunk is cast on the way). The array's ``base`` is
    ``out``. ``out`` may also be an ``(n, d)`` numpy array (a preloaded file's
    rows, ``_row_groups_into``), filled at its own dtype (the host widen happens
    in the copy) and returned."""
    if isinstance(out, np.ndarray):
        if out.shape != (n, d):
            raise ValueError(f"column {column!r}: a row group of {n} x {d} values where "
                             f"the footer gives {out.shape[0]} x {out.shape[1]} (ragged: "
                             "list lengths differ between row groups?)")
        dst = out
    else:
        dt = block_dtype(chunked.type)
        _check_slot_room(n * d * dt.itemsize, out)
        dst = np.ndarray((n, d), dt, buffer=out)
    row = 0
    for chunk in chunked.chunks:
        m = len(chunk)
        if not m:
            continue
        flat = chunk.flatten()
        if flat.null_count:
            raise ValueError(f"column {column!r} has null elements")
        src = flat.to_numpy(zero_copy_only=False).reshape(m, d)
        if src.dtype not in STORED_DTYPES and dst.dtype != np.float32:
            # e.g. float64 stored, float16 wanted: go through float32 first, like
            # the stream path (table_vectors' float32, then the host cast), so
            # both round the same way and the blocks are bit-identical
            src = src.astype(np.float32)
        np.copyto(dst[row:row + m], src, casting="same_kind")
        row += m
    return dst


def _out_for(table, column, dtype, out):
    """``out`` if the block can be decoded straight into it (no host cast to
    ``dtype`` would follow), else None."""
    if out is None or dtype is None:
        return out
    typ = table.schema.field(column).type
    if not is_vector_list(typ):
        return None  # table_vectors raises the TypeError
    return out if block_dtype(typ) == np.dtype(dtype) else None


# The verify reload decodes each row group in record batches of ~PRELOAD_BATCH_BYTES,
# copying each batch into the footer-sized array and dropping it. This bounds memory:
# a whole-file pq.read_table holds 3-6x the block, one read_row_group ~5x its row group.
PRELOAD_BATCH_BYTES = 4 << 20
# column chunks are streamed through a buffer this size (unbuffered, a whole chunk is
# read at once: 84 MB per decode thread on Pes2o-VE)
PRELOAD_BUFFER_BYTES = 8 << 20


def preload_workers(num_row_groups, threads=None):
    """Row groups ``_row_groups_into`` decodes at once: one per decode thread
    (default: Arrow's cpu pool), at most the file's row-group count."""
    return max(1, min(threads or pa.cpu_count(), num_row_groups))


def preload_batch_rows(d, itemsize, batch_bytes=PRELOAD_BATCH_BYTES):
    """Rows per record batch of ``_row_groups_into`` (at least 1)."""
    return max(1, batch_bytes // max(1, d * itemsize))


def arrow_array(shape, dtype):
    """A writable C-contiguous array over a page-aligned Arrow buffer.

    Gives reload blocks the same host memory as streamed blocks (4 KiB pages,
    page-aligned). ``np.empty`` differs: numpy requests huge pages for large arrays
    and its data starts 16 B past a page boundary."""
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    buf = pa.allocate_buffer(nbytes + mmap.PAGESIZE)
    lead = -buf.address % mmap.PAGESIZE
    return np.frombuffer(buf, np.uint8, nbytes, lead).view(dtype).reshape(shape)


def _row_groups_into(open_src, md, path, column, dtype, batch_bytes, buffer_size=0):
    """Decode ``column`` of a parquet file into a new ``(rows, d)`` ``arrow_array``.

    ``md`` is the file's metadata; ``open_src()`` opens a fresh input file over
    it (one per row group read). ``dtype`` None keeps the stored dtype, as
    ``table_vectors``. ``preload_workers`` row groups run at a time, each on one
    thread (``use_threads=False``), read in record batches of
    ``preload_batch_rows`` rows; each batch is copied straight into its rows
    (``table_vectors``' ``out`` path, with its null / ragged / d checks) and
    dropped. ``buffer_size`` > 0 streams each column chunk through a buffer of
    that size. Returns None when the footer cannot size the array (not a float
    list column): the caller then falls back to the whole-file read, which
    raises the proper error."""
    foot = footer_from_metadata(md, path, column)
    if foot.dim is None or foot.value_bits is None or foot.list_kind is None:
        return None
    stored = np.dtype(np.float16) if foot.value_bits == 16 else np.dtype(np.float32)
    x = arrow_array((md.num_rows, foot.dim), stored if dtype is None else dtype)
    rows = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
    starts = np.cumsum([0] + rows)
    batch = preload_batch_rows(foot.dim, foot.value_bits // 8, batch_bytes)

    def one(i):
        if not rows[i]:
            return
        at, end = int(starts[i]), int(starts[i + 1])
        with open_src() as src:
            pf = pq.ParquetFile(src, metadata=md, buffer_size=buffer_size, pre_buffer=False)
            for rb in pf.iter_batches(batch_size=batch, row_groups=[i], columns=[column],
                                      use_threads=False):
                n = rb.num_rows
                if at + n > end:
                    at += n
                    break  # more rows than the footer says
                table_vectors(pa.Table.from_batches([rb]), column, out=x[at:at + n])
                at += n
        if at != end:
            raise ValueError(f"{path}: row group {i} decoded {at - starts[i]} rows "
                             f"(or more), the footer says {rows[i]}")

    k = preload_workers(len(rows))
    if k == 1:
        for i in range(len(rows)):
            one(i)
        return x
    pool = ThreadPoolExecutor(max_workers=k, thread_name_prefix="rgdec")
    try:
        for fut in [pool.submit(one, i) for i in range(len(rows))]:
            fut.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return x


def read_file(fs, path, column, dtype=None, stats=None, size=None,
              mode=ARROW, range_concurrency=NATIVE_CONCURRENCY,
              range_bytes=NATIVE_PART_BYTES, footers=None, rec=None, out=None,
              batch_bytes=0, codec=None, extra_columns=(), extra=None):
    """Fetch and decode one Parquet file, optionally converting its dtype.

    ``native`` uses the Rust reader; ``arrow`` uses PyArrow for queries and
    verification. Records fetch, decode, and conversion times in ``stats``
    and ``rec``. Empty files raise an error.

    ``out`` allows decoding directly into a shared-memory ring slot when
    no host conversion is needed. ``batch_bytes`` enables parallel row-group
    decoding for verification.

    ``codec`` handles sparse and multivector data. ``extra_columns`` loads
    filter metadata alongside vectors, using a separate ETag-matched fetch
    in native mode."""
    if mode not in FILE_MODES:
        raise ValueError(f"read mode {mode!r} not in {FILE_MODES}")
    extra_columns = [c for c in extra_columns if c != column]
    if extra_columns:
        if extra is None:
            raise ValueError("extra_columns needs an `extra` dict to return them in")
        if codec is not None:
            raise ValueError("extra_columns with a sparse codec is not supported")
        if batch_bytes:
            # the reload's batch read cannot carry them, and a second read
            # could see another version of the object
            raise ValueError("extra_columns with the batch (preload) path is not supported")
        if mode == "native":
            # the native vector read cannot carry them: one more read of just these
            rec = rec if rec is not None else {}
            x = read_native(fs, path, column, dtype, stats, size, range_concurrency,
                            range_bytes, footers, rec, out)
            t = now()
            rec["payload_start"] = t
            # as nova-bf: the dense column through the native decoder, every
            # other column through rust_fetch's download + a pyarrow decode
            # (Store.read_columns_fetched), pinned to the dense read's ETag
            try:
                table, image, fetched, t1, t2 = _native_fetch_columns(
                    fs, path, extra_columns, range_concurrency, range_bytes,
                    pin_etag=rec.get("native_etag"))
            except NativeRefused as e:
                raise NativeRefused(f"{path}: --read-mode native refused the filter "
                                    f"columns ({e})") from e
            _release_image(image, table)
            del image
            extra["table"] = table
            if stats is not None:
                stats.add("payload_fetch", t1 - t)
                stats.add("payload_read_table", t2 - t1)
                stats.add("payload_fetched_bytes", fetched)
            rec.update(payload_fetch=t1 - t, payload_read_table=t2 - t1,
                       payload_path="native-fetch")
            t_end = now()
            if stats is not None:
                stats.add("payload_read", t_end - t)
            rec["payload_end"] = t_end
            rec["payload_read"] = t_end - t   # the filter-column read
            rec["w_end"] = t_end   # the whole read: payload included
            return x
    if codec is not None:
        if mode == "native":
            # nova-bf's path for a sparse column: rust_fetch downloads it,
            # pyarrow decodes (rust_fetch's own decoder is dense only)
            return _read_native_sparse(fs, path, column, dtype, stats, size,
                                       range_concurrency, range_bytes, footers, rec, out,
                                       codec)
        batch_bytes = 0
    if mode == "native":
        return read_native(fs, path, column, dtype, stats, size, range_concurrency,
                           range_bytes, footers, rec, out)
    t0 = now()
    looked_up = size is None
    if looked_up:
        size = fs.get_file_info(path).size
    if size == 0:
        raise ValueError(f"{path}: empty file")
    # the column's byte / row-group counts come from the footer; a file's
    # first read fetches it (one small GET), counted as fetch: the only part
    # of an arrow read that can be separated from decode
    foot = footers.get(path) if footers is not None else None
    if foot is None:
        foot = read_footer(fs, path, column)
        if footers is not None:
            footers[path] = foot
        looked_up = True
    nbytes, n_rg = foot.column_bytes, foot.num_row_groups
    md = pq.read_metadata(path, filesystem=fs) if batch_bytes else None
    t1 = now() if looked_up or md is not None else t0  # 0 when nothing was fetched
    # the batch path casts while it copies
    x = None if md is None else _row_groups_into(
        lambda: fs.open_input_file(path), md, path, column, dtype, batch_bytes,
        PRELOAD_BUFFER_BYTES)
    if x is None:
        table = pq.read_table(path, filesystem=fs, columns=[column, *extra_columns])
        t_rt = now()
        if table.num_rows == 0:
            raise ValueError(f"{path}: empty file")
        if extra_columns:
            extra["table"] = table.select(extra_columns)
            table = table.select([column])
        x = table_vectors(table, column, _out_for(table, column, dtype, out)) \
            if codec is None else _codec_decode(codec, table, column, dtype, out, stats)
        del table
    else:
        t_rt = now()
    t2 = now()
    t3 = t2
    if codec is not None:
        if codec.needs_widen(x, dtype):
            x = codec.widen(x, dtype)
            t3 = now()
    elif dtype is not None and x.dtype != dtype:
        x = np.ascontiguousarray(x.astype(dtype))
        t3 = now()
    read_table, convert = t_rt - t1, t2 - t_rt
    if stats is not None:
        stats.add("fetch", t1 - t0)
        stats.add("read_table", read_table)
        stats.add("convert", convert)
        stats.add("decode", read_table + convert)
        stats.add("widen", t3 - t2)
        stats.add("bytes", nbytes)
        stats.add("rows", x.shape[0] if codec is None else codec.rows(x))
        stats.add("files", 1)
        stats.add("row_groups", n_rg)
        stats.peak("max_row_groups", n_rg)
    if rec is not None:
        rec.update(w_start=t0, w_end=t3, fetch=t1 - t0, read_table=read_table,
                   convert=convert, widen=t3 - t2)
    return x


def _codec_decode(codec, table, column, dtype, out, stats):
    """``codec.decode``, passing ``stats`` to a codec with ``decode_stats`` set.

    That is sparse.ProjectingCodec, which times ``project`` inside ``convert``."""
    if getattr(codec, "decode_stats", False):
        return codec.decode(table, column, dtype, out, stats=stats)
    return codec.decode(table, column, dtype, out)


# ------------------------------------------------------- native (rust_fetch)

_NATIVE = {"pid": None, "clients": {}, "orphans": []}
_NATIVE_LOCK = threading.Lock()


def _reset_native_lock_in_child():
    """A forked child gets a fresh lock (the parent's may be held mid-fork)."""
    global _NATIVE_LOCK
    _NATIVE_LOCK = threading.Lock()


os.register_at_fork(after_in_child=_reset_native_lock_in_child)


class NativeRefused(ValueError):
    """``--read-mode native`` could not read a file the way nova-bf's native path does.

    Raised for rust_fetch ``UnsupportedLayout`` / ``CorruptParquet`` /
    ``MemoryError`` or the pyarrow schema guard. nova-bf would read that file
    with pyarrow instead; here a refused file fails the read."""


def native_import():
    """``rust_fetch`` (nova-bf's native I/O wheel), or a clear ImportError."""
    try:
        import rust_fetch
    except ImportError as e:
        raise ImportError(f"--read-mode native needs rust_fetch (nova-bf's native I/O "
                          f"wheel, python/nova-bf/rust_io): {e}") from e
    return rust_fetch


def native_url(fs, path):
    """``(url, region)`` rust_fetch reads ``path`` of ``fs`` by.

    ``s3://bucket/key`` with pyarrow's resolved region (as nova-bf ``Store``),
    or the absolute local path. Anything else (another filesystem, a custom S3
    endpoint) raises ValueError: nova-bf would read it with pyarrow, and native
    mode never silently reads another way."""
    kind = fs.type_name
    if kind == "s3":
        env = [k for k in NATIVE_ENDPOINT_ENV if os.environ.get(k)]
        if env:
            raise ValueError(f"--read-mode native: {env[0]} is set (custom S3 endpoint); "
                             "nova-bf reads such stores with pyarrow, not natively")
        return "s3://" + path, getattr(fs, "region", None) or None
    if kind == "local":
        return os.path.abspath(path), None
    raise ValueError(f"--read-mode native reads local paths and s3:// only, not {kind}")


def native_client(region, concurrency, part_bytes):
    """This process's rust_fetch Client for ``(region, concurrency, part_bytes)``.

    Made on first use, as nova-bf's ``_native_client``: one per process, shared
    by all its threads; ``concurrency`` caps in-flight GETs across all their
    calls. A Client inherited across ``fork()`` is unusable (rust_fetch refuses
    it), so a forked child builds its own and keeps the inherited ones
    referenced (dropping one in the child is not safe). Reader processes are
    spawned, so each simply makes its own."""
    rust_fetch = native_import()
    with _NATIVE_LOCK:
        if _NATIVE["pid"] != os.getpid():
            _NATIVE["orphans"].append(_NATIVE["clients"])
            _NATIVE["pid"], _NATIVE["clients"] = os.getpid(), {}
        key = (region, int(concurrency), int(part_bytes))
        client = _NATIVE["clients"].get(key)
        if client is None:
            client = rust_fetch.Client(region=region, concurrency=int(concurrency),
                                       part_size=int(part_bytes))
            _NATIVE["clients"][key] = client
        return client


def native_drop_clients():
    """Drop this process's Clients (their tokio runtimes and threads shut down).

    A run keeps its Client(s) to the end. Tests call this so Clients do not
    pile up in one long-lived process: each holds worker threads and their
    malloc arenas, and under ``ulimit -v`` dozens of them exhaust the address
    space."""
    with _NATIVE_LOCK:
        if _NATIVE["pid"] == os.getpid():
            _NATIVE["clients"] = {}


def native_warm_up(fs, path, concurrency, part_bytes):
    """Pay native mode's one-off costs outside the timed passes: the Client
    (tokio runtime, S3 credential / region resolution) and a first HEAD of
    ``path`` (TLS connection). Best effort, like ``_warm_up_fs``; a missing
    rust_fetch still raises (nothing could be read)."""
    native_import()
    try:
        url, region = native_url(fs, path)
        native_client(region, concurrency, part_bytes).head(url)
    except Exception:  # noqa: BLE001 - the timed read reports it
        pass


def native_schema_mismatch(foot, plan):
    """Why pyarrow would not read the column as the plan's plain ``(n_rows, dim)``
    float block, or None.

    nova-bf's ``pq.read_schema`` guard in ``_read_dense_native``, checked here
    against the pyarrow-parsed ``Footer`` (which honours ``ARROW:schema``)."""
    want = {"float16": ("halffloat", 16), "float32": ("float", 32)}.get(plan.dtype)
    if foot.list_kind is None:
        return f"pyarrow reads the column as {foot.value_type}, not a list"
    if want is None or (foot.value_type, foot.value_bits) != want:
        return f"pyarrow reads list<{foot.value_type}>, rust_fetch {plan.dtype}"
    if foot.num_rows != plan.n_rows:
        return f"pyarrow's footer has {foot.num_rows} rows, rust_fetch {plan.n_rows}"
    if plan.n_rows and foot.dim is not None and foot.dim != plan.dim:
        return f"pyarrow's footer gives d = {foot.dim}, rust_fetch {plan.dim}"
    return None


def _native_dense(fs, path, column, foot, concurrency, part_bytes, out, dtype):
    """Native dense read of one file; returns ``(x, plan, t_plan)``.

    ``plan`` is rust_fetch's ``dense_plan`` (a HEAD + the footer GET, ``size``
    None as nova-bf), checked against pyarrow's footer. Then ``read_dense_into``
    (ranged GETs of the column chunks + decode, pipelined, GIL released) writes
    straight into ``out`` (a ring slot) when no host cast follows, else into a
    new ``arrow_array``. ``t_plan`` is the ``now()`` stamp after the plan."""
    rust_fetch = native_import()
    url, region = native_url(fs, path)
    client = native_client(region, concurrency, part_bytes)
    refusals = (ValueError, MemoryError, rust_fetch.CorruptParquet)
    try:
        plan = client.dense_plan(url, column, None)
    except refusals as e:
        raise NativeRefused(f"{type(e).__name__}: {e}") from e
    t_plan = now()
    why = native_schema_mismatch(foot, plan)
    if why:
        raise NativeRefused(why)
    if plan.n_rows == 0:
        raise ValueError(f"{path}: empty file")
    stored = np.dtype(plan.dtype)
    shape = (plan.n_rows, plan.dim)
    if out is not None and (dtype is None or np.dtype(dtype) == stored):
        _check_slot_room(plan.n_rows * plan.dim * stored.itemsize, out, path)
        x = np.ndarray(shape, stored, buffer=out)  # x.base is out: nothing to publish
    else:
        x = arrow_array(shape, stored)
    try:
        client.read_dense_into(plan, x.reshape(-1).view(np.uint8), NATIVE_CHUNKS_IN_FLIGHT)
    except refusals as e:
        raise NativeRefused(f"{type(e).__name__}: {e}") from e
    return x, plan, t_plan

# Native fetch for columns not decoded directly by rust_fetch (sparse, IDs, text,
# filters). Downloads only the required Parquet chunks, then decodes with PyArrow.
# Unlike nova-bf, unsupported files raise an error instead of falling back to PyArrow.

_FETCH_PAD = 4096           # io.py: past each chunk (PARQUET-816 page-header slack)
_FETCH_COALESCE = 8 << 10   # io.py: merge ranges this close (pyarrow pre_buffer's hole limit)
_FETCH_TAIL = 1 << 20       # io.py: the first tail read (footer + metadata)
# io.py: the last images' pyarrow buffers stay referenced, so no Arrow worker
# thread ever drops the final reference to one (process-exit aborts)
_FETCH_KEEPALIVE = deque(maxlen=4)


def _fetch_image(client, url, size, pin):
    """io.py ``_fetch_image``: read the footer into a file-sized image.

    One tail read, plus a second read if the footer is longer than the tail.
    Returns ``(image, have_from)``, or None when the bytes are not a parquet footer."""
    if size < 12:
        return None
    image = _zeroed_image(size)
    if os.environ.get("NOVA_BF_RUST_FETCH_CANARY"):
        image.fill(0xA5)  # tests: make any read of an unfetched byte loud
    tail = min(size, _FETCH_TAIL)
    client.read_ranges_into(url, image, [(size - tail, tail, size - tail)], if_match=pin)
    if bytes(image[size - 4:]) != b"PAR1":
        return None
    flen = int.from_bytes(bytes(image[size - 8:size - 4]), "little")
    if flen + 8 > size:
        return None
    have_from = size - tail
    if flen + 8 > tail:
        start = size - 8 - flen
        client.read_ranges_into(url, image, [(start, size - tail - start, start)], if_match=pin)
        have_from = start
    return image, have_from


def _zeroed_image(size):
    """io.py ``_zeroed_image``: a zeroed, writable uint8 array.

    Above 1 MiB it is an anonymous private mmap (pages allocated only when
    written) with huge pages off, so the unfetched parts of the file cost no memory."""
    if size < (1 << 20):
        return np.zeros(size, dtype=np.uint8)
    m = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    if hasattr(mmap, "MADV_NOHUGEPAGE"):
        m.madvise(mmap.MADV_NOHUGEPAGE)
    return np.frombuffer(m, dtype=np.uint8)


def _release_image(image, table):
    """io.py ``_release_image``: give the image's pages back (MADV_DONTNEED)
    unless an Arrow buffer of ``table`` still points into it."""
    base = image.ctypes.data
    end = base + image.nbytes
    for col in table.columns:
        for chunk in col.chunks:
            stack = [chunk]
            while stack:
                arr = stack.pop()
                for b in arr.buffers():
                    if b is not None and b.size and base <= b.address < end:
                        return
                stack.extend(getattr(arr, "children", None) or [])
                dct = getattr(arr, "dictionary", None)
                if dct is not None:
                    stack.append(dct)
    m = image.base
    m = getattr(m, "obj", m)  # np.frombuffer(mmap) -> memoryview -> mmap
    if isinstance(m, mmap.mmap) and hasattr(mmap, "MADV_DONTNEED"):
        m.madvise(mmap.MADV_DONTNEED)


def _leaf_count(t):
    """io.py ``_leaf_count``: parquet leaf columns of an Arrow type."""
    if isinstance(t, pa.BaseExtensionType):  # incl. C++ canonical (uuid, json, opaque)
        return _leaf_count(t.storage_type)
    if pa.types.is_struct(t):
        return sum(_leaf_count(t.field(i).type) for i in range(t.num_fields))
    if pa.types.is_map(t):
        return _leaf_count(t.key_type) + _leaf_count(t.item_type)
    if (pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t)
            or pa.types.is_list_view(t) or pa.types.is_large_list_view(t)):
        return _leaf_count(t.value_type)
    return 1


def _column_ranges(image, size, columns, have_from=None):
    """io.py ``_column_ranges``: the selected columns' chunk ranges as
    ``(offset, length, dest)`` triples.

    From rust_fetch.chunk_spans, padded, coalesced, the fetched footer
    excluded. None when the metadata is ambiguous or unsupported (io.py then
    reads with pyarrow)."""
    end = size if have_from is None else have_from
    try:
        pf = pq.ParquetFile(pa.BufferReader(pa.py_buffer(image)))
        md, arrow = pf.metadata, pf.schema_arrow
        names = arrow.names
        counts = [_leaf_count(f.type) for f in arrow]
        if sum(counts) != md.num_columns or len(set(names)) != len(names):
            return None
        want = set(columns)
        if not want <= set(names):
            return None
        leaves, j = [], 0
        for name, c in zip(names, counts):
            if name in want:
                leaves.extend(range(j, j + c))
            j += c
        # Chunk offsets come from rust_fetch, not pyarrow's RowGroupMetaData
        # .column(): on some malformed size statistics that accessor lets a
        # C++ exception reach std::terminate and aborts the whole process
        # (a plain pyarrow read raises a clean error on the same file).
        # chunk_spans refuses those footers, so we decline instead.
        rust_fetch = native_import()
        flen = int.from_bytes(bytes(image[size - 8:size - 4]), "little")
        if flen <= 0 or flen > size - 12:
            return None
        footer = bytes(image[size - 8 - flen:size - 8])
        spans = []
        for start, dpo, length in rust_fetch.chunk_spans(footer, leaves):
            if dpo is not None and 0 < dpo < start:
                start = dpo
            if start < 0 or length < 0 or start + length > size:
                return None
            s0, e0 = start, min(end, start + length + _FETCH_PAD)
            if s0 < e0:
                spans.append((s0, e0))
    except Exception:  # noqa: BLE001 - io.py: any doubt declines
        return None
    if not spans:
        return []
    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1] + _FETCH_COALESCE:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e - s, s) for s, e in merged]


def _native_fetch(fs, path, column, concurrency, part_bytes):
    """One column: ``_native_fetch_columns(..., [column])``."""
    return _native_fetch_columns(fs, path, [column], concurrency, part_bytes)


def _native_fetch_columns(fs, path, columns, concurrency, part_bytes, pin_etag=None):
    """nova-bf's ``Store.read_columns_fetched``; returns
    ``(table, image, fetched_bytes, t_fetched, t_read)``.

    HEAD (size, ETag), footer into the image, the columns' chunk ranges in one
    ETag-pinned ``read_ranges_into``, then ``pq.read_table`` from the image.
    Every case where io.py would fall back to pyarrow raises ``NativeRefused``
    instead; transport errors and missing files raise as they are, because a
    benchmark must never silently switch readers. ``pin_etag`` (filter columns after
    a dense read) refuses an object whose ETag changed, so filter columns are never
    paired with another version's vectors."""
    rust_fetch = native_import()
    if any(c.startswith("__") for c in columns) or \
            any("=" in seg for seg in path.split("/")[:-1]):
        raise NativeRefused("a __ column or a hive k=v directory: io.py reads it with "
                            "pyarrow (dataset semantics)")
    url, region = native_url(fs, path)
    client = native_client(region, concurrency, part_bytes)
    refusals = (ValueError, MemoryError, rust_fetch.CorruptParquet)
    image = None
    try:
        size, pin = client.head(url)
        if pin_etag is not None and pin != pin_etag:
            raise NativeRefused(f"object changed since the dense read (ETag {pin_etag!r} "
                                f"-> {pin!r})")
        got = _fetch_image(client, url, size, pin)
        if got is None:
            raise NativeRefused("not a parquet footer")
        image, have_from = got
        ranges = _column_ranges(image, size, list(columns), have_from)
        if ranges is None:
            try:   # a missing column is the arrow mode's KeyError, not a refusal
                names = pq.ParquetFile(pa.BufferReader(pa.py_buffer(image))).schema_arrow.names
            except Exception:  # noqa: BLE001 - unreadable metadata: refuse below
                names = None
            missing = [c for c in columns if names is not None and c not in names]
            if missing:
                _release_image(image, pa.table({}))
                raise KeyError(f"{path}: no column {missing[0]!r}")
            raise NativeRefused("column ranges declined (ambiguous or unsupported metadata)")
        if ranges:
            client.read_ranges_into(url, image, ranges, if_match=pin)
    except BaseException as e:
        # whatever ends the fetch, give the image's pages back at once (the
        # error's traceback may keep this frame alive for a while)
        if image is not None:
            _release_image(image, pa.table({}))
        if isinstance(e, NativeRefused):
            raise
        if isinstance(e, refusals):
            raise NativeRefused(f"{type(e).__name__}: {e}") from e
        if isinstance(e, OSError):
            if "object changed" in str(e):  # io.py: falls back to pyarrow
                raise NativeRefused(f"object changed during the read: {e}") from e
            if e.errno == errno.ENOMEM:     # the image mmap (io.py: falls back to pyarrow)
                raise NativeRefused(f"could not map the file-sized image: {e}") from e
        raise
    t_fetched = now()
    fetched = (size - have_from) + sum(n for _, n, _ in ranges)  # footer reads + chunks
    buf = pa.py_buffer(image)
    try:
        table = pq.read_table(pa.BufferReader(buf), columns=list(columns), use_threads=True)
    except Exception as e:  # noqa: BLE001 - io.py: any failure falls back
        _release_image(image, pa.table({}))  # nothing decoded from it: give its pages back
        raise NativeRefused(f"pyarrow could not read the fetched image: {e}") from e
    finally:
        _FETCH_KEEPALIVE.append(buf)
    return table, image, fetched, t_fetched, now()


def _read_native_sparse(fs, path, column, dtype, stats, size, concurrency, part_bytes,
                        footers, rec, out, codec):
    """``read_native`` for a codec column (sparse / multivector): rust_fetch download,
    pyarrow decode, then ``codec.decode`` into the packed block (``out`` unless widened).

    Timing: ``fetch`` (= ``native_read``) = HEAD + footer + chunk GETs and the range
    planning; ``read_table`` = pyarrow decode from the image; ``convert`` = footer
    facts, image release and ``codec.decode``; ``widen`` = ``codec.widen``. ``bytes``
    is the column's compressed size; ``fetched_bytes`` what was actually downloaded
    (a file under 1 MiB is fetched whole). A refused file raises ``NativeRefused``."""
    t0 = now()
    if size is None:
        size = fs.get_file_info(path).size
    if size == 0:
        raise ValueError(f"{path}: empty file")
    try:
        table, image, fetched, t1, t_rt = _native_fetch(fs, path, column, concurrency,
                                                        part_bytes)
    except NativeRefused as e:
        raise NativeRefused(f"{path}: --read-mode native refused this file ({e}); "
                            "nova-bf would read it with pyarrow") from e
    if table.num_rows == 0:
        _release_image(image, pa.table({}))
        raise ValueError(f"{path}: empty file")
    # footer facts every mode reports (row groups; ``bytes`` = the column's
    # compressed bytes): from the Footer cache (ParquetSource.start), else
    # parsed from the fetched image's footer; timed in ``convert``
    foot = footers.get(path) if footers is not None else None
    if foot is None:
        try:
            foot = footer_from_metadata(pq.read_metadata(pa.BufferReader(pa.py_buffer(image))),
                                        path, column)
        except BaseException:
            _release_image(image, pa.table({}))
            raise
    _release_image(image, table)
    x = _codec_decode(codec, table, column, dtype, out, stats)
    del table, image
    t2 = t3 = now()
    if codec.needs_widen(x, dtype):
        x = codec.widen(x, dtype)
        t3 = now()
    read_table, convert = t_rt - t1, t2 - t_rt
    if stats is not None:
        stats.add("fetch", t1 - t0)
        stats.add("native_read", t1 - t0)
        stats.add("read_table", read_table)
        stats.add("convert", convert)
        stats.add("decode", read_table + convert)
        stats.add("widen", t3 - t2)
        stats.add("bytes", foot.column_bytes)
        stats.add("fetched_bytes", fetched)
        stats.add("rows", codec.rows(x))
        stats.add("files", 1)
        stats.add("native_fallbacks", 0)
        stats.add("row_groups", foot.num_row_groups)
        stats.peak("max_row_groups", foot.num_row_groups)
    if rec is not None:
        rec.update(w_start=t0, w_end=t3, fetch=t1 - t0, read_table=read_table,
                   convert=convert, widen=t3 - t2, native_read=t1 - t0,
                   read_path="native-fetch")
    return x


def read_native(fs, path, column, dtype=None, stats=None, size=None,
                concurrency=NATIVE_CONCURRENCY, part_bytes=NATIVE_PART_BYTES,
                footers=None, rec=None, out=None):
    """``read_file`` for ``mode="native"``: nova-bf's ``Store._read_dense_native``.

    Timing (same keys as arrow mode): ``fetch`` = uncached footer read + ``dense_plan``
    (HEAD + footer GET); ``read_table`` (= ``native_read``) = ``read_dense_into``, the
    chunk GETs and decode, pipelined inside rust_fetch so not separable; ``convert``
    ~0 (decoded in place); ``widen`` = host cast. Refused files, transport errors and
    missing files raise: a run never reads another way."""
    t0 = now()
    if size is None:
        size = fs.get_file_info(path).size
    if size == 0:
        raise ValueError(f"{path}: empty file")
    foot = footers.get(path) if footers is not None else None
    if foot is None:
        foot = read_footer(fs, path, column)  # pyarrow's view, for the schema guard
        if footers is not None:
            footers[path] = foot
    try:
        x, plan, t1 = _native_dense(fs, path, column, foot, concurrency, part_bytes, out,
                                    dtype)
    except NativeRefused as e:
        raise NativeRefused(f"{path}: --read-mode native refused this file ({e}); "
                            "nova-bf would read it with pyarrow") from e
    t_rt = now()
    t2 = t3 = now()
    if dtype is not None and x.dtype != dtype:
        x = np.ascontiguousarray(x.astype(dtype))
        t3 = now()
    read_table, convert = t_rt - t1, t2 - t_rt
    if stats is not None:
        stats.add("fetch", t1 - t0)
        stats.add("read_table", read_table)
        stats.add("native_read", read_table)
        stats.add("convert", convert)
        stats.add("decode", read_table + convert)
        stats.add("widen", t3 - t2)
        stats.add("bytes", plan.column_bytes)
        stats.add("rows", x.shape[0])
        stats.add("files", 1)
        stats.add("native_fallbacks", 0)
        stats.add("row_groups", plan.num_row_groups)
        stats.peak("max_row_groups", plan.num_row_groups)
    if rec is not None:
        rec.update(w_start=t0, w_end=t3, fetch=t1 - t0, read_table=read_table,
                   convert=convert, widen=t3 - t2, native_read=read_table,
                   read_path="native", native_etag=plan.etag)
    return x


# ------------------------------------------------- shared-memory block ring

# how reader processes share the decode thread total: split = shares sum to it;
# full = each reader gets all of it (oversubscribed when several decode at once)
READER_POOLS = ("split", "full")
SHM_DIR = "/dev/shm"
SHM_PREFIX = "nova-stream-"
# MAP_POPULATE: readers map ring slots with page tables filled in at start-up, so the
# first copy into a slot takes no per-page faults inside the timed publish
_MAP_POPULATE = getattr(mmap, "MAP_POPULATE", 0)
# Python >= 3.13: an mmap need not keep its fd open (trackfd=False). Before 3.13 every
# process holds one fd per ring slot, so RLIMIT_NOFILE must cover the ring.
_MMAP_KW = {"trackfd": False} if sys.version_info >= (3, 13) else {}
# Most tasks in flight (--prefetch). The parent writes every in-flight task (16 B) to
# the task pipe before reading any reply, so the pipe must hold them all or the
# parent and readers deadlock. 1024 x 16 B = 16 KiB fits a default 64 KiB pipe;
# _ProcReaders.start still checks (and raises) the real capacity, since a user over
# /proc/sys/fs/pipe-user-pages-soft gets 4-8 KiB pipes.
MAX_PREFETCH = 1024
_F_GETPIPE_SZ = getattr(fcntl, "F_GETPIPE_SZ", None)  # Linux, Python >= 3.10
_F_SETPIPE_SZ = getattr(fcntl, "F_SETPIPE_SZ", None)


def split_threads(total, readers):
    """``total`` threads split across ``readers``: ``total // readers`` each,
    the first ``total % readers`` readers one more, so they sum to ``total``
    (e.g. 7 over 3: ``[3, 2, 2]``). A reader needs at least one thread, so a
    ``total`` smaller than ``readers`` gives every reader 1 (``readers`` in
    all, more than asked)."""
    readers = max(1, readers)
    base, extra = divmod(max(0, total), readers)
    return [max(1, base + (i < extra)) for i in range(readers)]


def reader_pool_threads(cpu_thread_count, io_thread_count, readers, pool="split"):
    """Arrow pool sizes ``(cpu, io)`` per reader process, two lists indexed by reader.

    ``split`` (default): the readers together get what one shared pool would. Decode
    total = ``cpu_thread_count`` or, at 0, the usable CPUs; IO total =
    ``io_thread_count`` or, at 0, zeros (each reader then splits Arrow's own default
    itself). ``full``: every reader gets the whole decode total; IO stays split (it
    serves only the start-up footer read, not the timed GETs)."""
    if pool not in READER_POOLS:
        raise ValueError(f"reader pool {pool!r} not in {READER_POOLS}")
    readers = max(1, readers)
    total = cpu_thread_count if cpu_thread_count and cpu_thread_count > 0 \
        else usable_cpu_count()
    io = split_threads(io_thread_count, readers) if io_thread_count and io_thread_count > 0 \
        else [0] * readers
    cpu = [total] * readers if pool == "full" else split_threads(total, readers)
    return cpu, io


def pool_split_str(counts):
    """Per-reader pool sizes in the CSV's compact form: ``<readers>x<threads>``
    groups joined by ``+``, larger pools first (``[3, 2, 2]`` -> ``1x3+2x2``;
    an even split is one group, ``4x2``). None if any size is unknown."""
    if not counts or any(c is None for c in counts):
        return None
    groups = Counter(counts)
    return "+".join(f"{groups[t]}x{t}" for t in sorted(groups, reverse=True))


def _shm_path(name):
    """Path of shared-memory segment ``name`` under /dev/shm."""
    return os.path.join(SHM_DIR, name)


def _unlink(name):
    """Remove segment ``name``; a missing one is fine."""
    try:
        os.unlink(_shm_path(name))
    except FileNotFoundError:
        pass


def _sweep_shm(prefix):
    """Remove every segment named ``prefix*`` (one reader pool's ring: parent
    pid + random token)."""
    for p in glob.glob(_shm_path(glob.escape(prefix) + "*")):
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass


def shm_segments(prefix=SHM_PREFIX):
    """Names of the live segments starting with ``prefix`` (tests / debugging)."""
    return sorted(os.path.basename(p) for p in glob.glob(_shm_path(glob.escape(prefix) + "*")))


def shm_free_bytes():
    """Bytes an unprivileged process can still allocate in /dev/shm."""
    st = os.statvfs(SHM_DIR)
    return st.f_bavail * st.f_frsize


def _madvise_huge(mm):
    """Best effort ``MADV_HUGEPAGE``; True if the advice was accepted.

    On /dev/shm (a tmpfs mount) it only takes effect when the mount allows it
    (``huge=advise``, or ``always`` / ``within_size``, which need no advice);
    the default ``huge=never`` keeps 4 KiB pages."""
    adv = getattr(mmap, "MADV_HUGEPAGE", None)
    if adv is None:
        return False
    try:
        mm.madvise(adv)
        return True
    except OSError:
        return False


def create_slot(name, nbytes):
    """Create, map and pre-fault shared-memory segment ``name`` of ``nbytes``.

    The segment is ``/dev/shm/<name>`` (what ``shm_open`` /
    ``multiprocessing.shared_memory`` create on Linux). After huge-page advice,
    one write per page allocates every tmpfs page now, before the timed passes,
    so no later copy into it pays for page allocation. Made by hand rather than
    with ``SharedMemory`` to stay out of multiprocessing's resource tracker (the
    pool owns the cleanup). The caller must check /dev/shm has room first
    (``_BlockRing``): a tmpfs that fills up while pages are touched gives
    SIGBUS, not an exception. Returns the parent's mapping; on any failure the
    segment is removed."""
    fd = os.open(_shm_path(name), os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.ftruncate(fd, nbytes)
        mm = mmap.mmap(fd, nbytes, flags=mmap.MAP_SHARED,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE, **_MMAP_KW)
    except BaseException:
        os.close(fd)
        _unlink(name)
        raise
    os.close(fd)
    try:
        _madvise_huge(mm)  # before the faults, or it cannot apply
        np.frombuffer(mm, np.uint8)[::mmap.PAGESIZE] = 0
    except BaseException:
        mm.close()
        _unlink(name)
        raise
    return mm


def map_slot(name):
    """A reader's mapping of an existing ring slot, page tables populated."""
    fd = os.open(_shm_path(name), os.O_RDWR)
    try:
        size = os.fstat(fd).st_size
        mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED | _MAP_POPULATE,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE, **_MMAP_KW)
    finally:
        os.close(fd)
    _madvise_huge(mm)
    return mm


def publish_to_slot(x, mm, path=""):
    """Copy block ``x`` (C-contiguous) into the start of slot mapping ``mm``:
    one memcpy (numpy, GIL released) into pages that are already allocated
    and mapped."""
    _check_slot_room(x.nbytes, mm, path)
    dst = np.ndarray(x.shape, x.dtype, buffer=mm)
    np.copyto(dst, x)
    del dst  # release the export


def buffer_address(buf):
    """The start address of a writable buffer (a ring slot's mmap), for
    cudaHostRegister. The numpy export is dropped before returning."""
    a = np.frombuffer(buf, np.uint8)
    addr = a.ctypes.data
    del a
    return addr


class _BlockRing:
    """The persistent shared-memory slots blocks travel through.

    ``n`` slots of ``slot_bytes`` (the run's largest block), created and pre-faulted
    before the passes and reused for every file. ``view`` hands a slot out as a
    zero-copy array whose finalizer frees the slot once the array and everything
    derived from it are gone, so a slot is never overwritten while the consumer can
    see it. ``prefetch + 1`` slots suffice when the consumer drops each block before
    the next; a consumer that keeps blocks makes ``take`` grow the ring (``grown``).
    The free list is a deque because finalizers run on any thread. /dev/shm room is
    checked up front (a full tmpfs gives SIGBUS, not an error)."""

    def __init__(self, prefix, slot_bytes, n, pin=None):
        self.prefix, self.slot_bytes = prefix, max(1, int(slot_bytes))
        self.mms, self.free, self.grown = [], deque(), 0
        # --h2d pinned: each slot is page-locked once after creation and unregistered
        # before its mapping can go (close(), or this finalizer, which holds the mappings)
        self.pin, self.pinned = pin, []
        self._fin = weakref.finalize(self, _BlockRing._unpin, pin, self.pinned, self.mms)
        self._check_room(n)
        try:
            for _ in range(n):
                self._add()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _unpin(pin, pinned, mms):
        """Unregister every pinned slot (once: the list is emptied first, so
        a second call, or one cut short, never unregisters twice). ``mms``
        is only held so no slot is unmapped before this ran."""
        if pin is None or not pinned:
            return
        addrs = list(pinned)
        pinned.clear()
        pin.unregister(addrs)

    def name(self, i):
        """Segment name of slot ``i``."""
        return f"{self.prefix}slot{i}"

    def _check_room(self, n):
        need, have = n * self.slot_bytes, shm_free_bytes()
        if need > have:
            raise OSError(errno.ENOSPC,
                          f"shared memory: the block ring needs {n} x {self.slot_bytes} B "
                          f"= {need / 2**30:.2f} GiB in {SHM_DIR}, only {have / 2**30:.2f} "
                          "GiB free; /dev/shm must hold (prefetch + 1) x the largest "
                          "block (Docker: --shm-size)")

    def _add(self):
        i = len(self.mms)
        self.mms.append(create_slot(self.name(i), self.slot_bytes))
        if self.pin is not None:
            # a slot that grows the ring is pinned too (run_pass never grows it)
            addr = buffer_address(self.mms[i])
            self.pin.register(addr, self.slot_bytes)
            self.pinned.append(addr)
        self.free.append(i)
        return i

    def __len__(self):
        return len(self.mms)

    def take(self):
        """A free slot; grows the ring by one when none is free."""
        try:
            return self.free.popleft()
        except IndexError:
            pass
        self._check_room(1)
        self.grown += 1
        self._add()
        return self.free.pop()

    def put(self, slot):
        self.free.append(slot)

    def view(self, slot, shape, dtype):
        """Array over ``slot``; the slot goes back on the free list when the array is gone."""
        x = np.ndarray(shape, dtype, buffer=self.mms[slot])
        fin = weakref.finalize(x, self.free.append, slot)
        fin.atexit = False
        return x

    def view_parts(self, slot, parts):
        """Several arrays over one slot, ``parts`` = [(offset, shape, dtype)].

        All are views of one base array that carries the free-list finalizer,
        so the slot comes back only when every part (and anything derived
        from them) is gone."""
        base = np.ndarray((self.slot_bytes,), np.uint8, buffer=self.mms[slot])
        fin = weakref.finalize(base, self.free.append, slot)
        fin.atexit = False
        out = []
        for offset, shape, dtype in parts:
            dtype = np.dtype(dtype)
            n = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            out.append(base[offset:offset + n].view(dtype).reshape(shape))
        del base
        return out

    def close(self):
        try:
            self._fin()  # unpin before any mapping can go (no-op when not pinned)
        finally:
            for i in range(len(self.mms)):
                _unlink(self.name(i))
            self.mms = []  # unmapped when the last array over each is gone
            self.free.clear()


# ------------------------------------------------------------ reader processes

def _merge_stats(stats, d):
    """Fold a worker's ``Stages.as_dict()`` into ``stats`` (``max_*`` keys are
    peaks, the rest sums)."""
    for key, v in d.items():
        if key.startswith("max_"):
            stats.peak(key, v)
        else:
            stats.add(key, v)


# One task on the shared task pipe: (tid, ring slot), 16 bytes; tid < 0 = stop.
# Far below PIPE_BUF (4096), so the parent's os.write of a record is atomic and
# each reader's os.read of exactly one record gets one whole record, never part
# of one: no lock between the readers (see _ProcReaders).
_TASK = struct.Struct("<qq")


def _recv_task(fd):
    """The next ``(tid, slot)`` from the shared task pipe, or None on EOF (the
    parent is gone: nobody else holds the write end)."""
    b = os.read(fd, _TASK.size)
    while b and len(b) < _TASK.size:  # never seen (atomic records); be safe
        more = os.read(fd, _TASK.size - len(b))
        if not more:
            return None
        b += more
    return _TASK.unpack(b) if b else None


def _warm_up_reads():
    """Pay a reader's one-off read costs at start-up, outside the timed passes.

    ``pq.read_table`` imports ``pyarrow.dataset`` lazily (~0.1 s) and the first
    parquet read / decode / ``table_vectors`` warms Arrow's code paths. Without
    this they land in each reader's first timed read (~110 ms vs ~1.5 ms for
    later reads of a tiny file on a dev box)."""
    import pyarrow.dataset  # noqa: F401 - imported lazily by pq.read_table
    buf = pa.BufferOutputStream()
    pq.write_table(pa.table({"v": pa.array([[0.0, 1.0]] * 4, pa.list_(pa.float32()))}), buf)
    raw = buf.getvalue()
    pq.read_metadata(pa.BufferReader(raw))
    table_vectors(pq.read_table(pa.BufferReader(raw), columns=["v"]), "v")


def _warm_up_fs(fs, path):
    """Pay a reader's one-off filesystem costs at start-up, outside the timed passes.

    One footer read of a corpus file through this process's own (pickled)
    ``fs``: on S3, credential resolution, client creation and the first TLS
    connection happen on first use, which would otherwise be the reader's
    first timed read. Best effort: a failure is left for the timed read of
    that file to report."""
    try:
        with fs.open_input_file(path) as f:
            pq.read_metadata(f)
    except Exception:  # noqa: BLE001 - the real read reports it
        pass


def _reader_main(tasks, conn, fs, read_fn, cpu_threads, io_threads, index, readers,
                 prefix, n_slots, files, kw):
    """Main loop of reader process ``index`` of ``readers``.

    Start-up (untimed): size its Arrow pools, warm up reads and the filesystem, map the
    ring, send ``ready``. Then per task from the shared pipe: send ``("start", tid)`` (so
    the parent can name the file if this reader dies), read ``files[tid]`` straight into
    the assigned slot (or copy it there after a host widen), and reply with shape /
    dtype, stats, the per-file ``rec`` and the footer used. ``rec`` also gets the
    process's user / sys CPU seconds for the task (``getrusage`` delta, all threads;
    one task at a time, so it is this file's). Failures reply with the exception.

    SIGINT is ignored: the parent owns cleanup. If the parent is gone (EOF, broken
    pipe, re-parented), it died without cleaning up, so the reader removes the ring's
    segments itself and exits."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ppid = os.getppid()
    if not io_threads:  # its share of Arrow's default IO pool, as this fresh process sees it
        io_threads = split_threads(pa.io_thread_count(), readers)[index]
    configure_arrow_pools(cpu_threads, io_threads)
    _warm_up_reads()
    if files:
        _warm_up_fs(fs, files[0][0])
        if kw.get("mode") == "native":  # this process's own Client, outside the passes
            native_warm_up(fs, files[0][0], kw["range_concurrency"], kw["range_bytes"])
    read_fn = read_fn or read_file
    # filtered search: each reader builds its own filter at start-up, so query-side
    # prep and self-checks run outside the timed passes
    t_flt = now()
    flt = kw["filter_job"].build() if kw.get("filter_job") is not None else None
    flt_setup = now() - t_flt if flt is not None else None   # -> filter_setup_s
    flt_fields = kw["filter_job"].fields if flt is not None else ()
    slots = {i: map_slot(f"{prefix}slot{i}") for i in range(n_slots)}
    vec_regions = {}  # slot -> (memoryview of its vector region, its address)
    foot_cache = {}   # path -> the latest footer this reader read (a rewrite refreshes it)
    conn.send(("ready", os.getpid(), pa.cpu_count(), pa.io_thread_count(), flt_setup))
    fd = tasks.fileno()
    global _TASK_SLOT
    while True:
        try:
            task = _recv_task(fd)
        except OSError:
            task = None
        if task is None or os.getppid() != ppid:
            _sweep_shm(prefix)  # parent gone without its cleanup; tasks are moot
            return
        tid, slot = task
        if tid < 0:
            return
        path, size, footer = files[tid]
        footer = foot_cache.get(path, footer)
        try:
            conn.send(("start", tid))
        except (OSError, EOFError):
            _sweep_shm(prefix)
            return
        c0 = resource.getrusage(resource.RUSAGE_SELF)
        try:
            if slot not in slots:  # the ring grew after this reader started
                slots[slot] = map_slot(f"{prefix}slot{slot}")
            _TASK_SLOT = slots[slot]  # the slot this task writes (tests / debugging)
            stats, rec = Stages(), {}
            footers = {path: footer} if footer is not None else {}
            extra = {} if flt is not None else None
            fkw = {"extra_columns": flt_fields, "extra": extra} if flt is not None else {}
            # With a filter the slot ends in a mask region: the read may only
            # write the vector region, so a block that does not fit (a file
            # that grew after start(), e.g. rewritten and re-read) fails the
            # room check loudly instead of spilling into the masks.
            if flt is None:
                out = slots[slot]
            else:
                if slot not in vec_regions:
                    mv = memoryview(slots[slot])[:kw["mask_offset"]]
                    vec_regions[slot] = (mv, buffer_address(mv))
                out = vec_regions[slot][0]
            x = read_fn(fs, path, kw["column"], kw["dtype"], stats, size, kw["mode"],
                        kw["range_concurrency"], kw["range_bytes"], footers, rec=rec,
                        out=out, **fkw)
            if footers.get(path) is not None:
                foot_cache[path] = footers[path]
            t = now()
            # decoded straight into the slot? (numpy resolves a memoryview
            # buffer to the mmap, so with a filter compare addresses)
            in_place = (x.base is out) if flt is None else \
                (x.__array_interface__["data"][0] == vec_regions[slot][1])
            if not in_place:  # not decoded into the slot (host widen)
                publish_to_slot(x, out, path)
            rec["publish_end"] = now()
            rec["publish"] = rec["publish_end"] - t
            shape, dtype = x.shape, x.dtype.str
            n_rows = x.shape[0]
            del x
            if flt is not None:
                from bench.data.filters import reader as filtered
                table = extra.pop("table")
                if table.num_rows != n_rows:
                    raise RuntimeError(f"{path}: {table.num_rows} filter rows vs {n_rows} vectors")
                rec["filter_start"] = now()
                masks, info = filtered.evaluate(flt, table, stats, rec)
                del table
                if kw.get("filter_compact") is not None:
                    # gather / union: move the kept rows to the front of the
                    # slot's vector region (in place) and their file rows into
                    # its keep region; the reply's shape is the compacted block's
                    vec = np.ndarray(shape, np.dtype(dtype), buffer=out)
                    vec, masks, info, comp = filtered.compact_block(
                        kw["filter_compact"], vec, masks, info, stats, rec, in_place=True)
                    shape = vec.shape
                    del vec
                    t = now()
                    keep = comp.keep
                    if keep.nbytes > kw["keep_capacity"]:
                        raise RuntimeError(f"{path}: {len(keep)} kept rows exceed the slot's "
                                           "keep region (the file grew after start()?)")
                    if len(keep):
                        dst = np.ndarray(keep.shape, np.int64, buffer=slots[slot],
                                         offset=kw["keep_offset"])
                        np.copyto(dst, keep)
                        del dst
                    info["compact"] = (comp.mode, len(keep), comp.n_rows, comp.added)
                    del keep, comp
                    dt = now() - t
                    rec["filter_compact"] += dt
                    stats.add("filter_compact", dt)
                if not filtered.publish_masks(masks, slots[slot], kw["mask_offset"],
                                              kw["mask_capacity"], path, stats, rec):
                    info["masks_inline"] = np.ascontiguousarray(masks)  # overflow: in the reply
                rec["filter_end"] = now()
                filtered.finish_timing(stats, rec)
                rec["filter_info"] = info
                del masks
            c1 = resource.getrusage(resource.RUSAGE_SELF)
            rec["cpu_user"] = c1.ru_utime - c0.ru_utime
            rec["cpu_sys"] = c1.ru_stime - c0.ru_stime
            reply = ("ok", tid, shape, dtype, stats.as_dict(), rec, footers.get(path))
        except BaseException as e:  # noqa: BLE001 - shipped to the parent
            reply = ("err", tid, e, traceback.format_exc())
        try:
            _send_reply(conn, reply)
        except (OSError, EOFError):
            _sweep_shm(prefix)  # parent gone without its cleanup
            return


def _send_reply(conn, reply):
    """``conn.send(reply)``; an exception that does not pickle is sent as a
    RuntimeError with its type and message (pipe errors propagate)."""
    try:
        conn.send(reply)
    except (OSError, EOFError):
        raise
    except Exception:  # noqa: BLE001 - unpicklable exception
        _, tid, e, tb = reply
        conn.send(("err", tid, RuntimeError(f"{type(e).__name__}: {e}"), tb))


_TASK_SLOT = None  # in a reader process: the ring slot mapping the current task writes


class ReaderDied(RuntimeError):
    """A reader process exited while it had a file in flight (or was handed
    one)."""


@contextlib.contextmanager
def _main_not_reimported():
    """Swap ``__main__`` for a bare module while the readers are spawned.

    spawn re-runs the parent's ``__main__`` in every child (as ``__mp_main__``):
    with the harness CLI as main that imports faiss (a GPU build on the node),
    torch, ... in every reader, for nothing. A reader needs only this module and
    its ``read_fn``'s. So a ``read_fn`` must live in an importable module
    (``__main__``'s functions cannot be pickled meanwhile)."""
    main = sys.modules.get("__main__")
    sys.modules["__main__"] = types.ModuleType("__main__")
    try:
        yield
    finally:
        if main is None:
            del sys.modules["__main__"]
        else:
            sys.modules["__main__"] = main


def ensure_pipe_room(fd, nbytes):
    """Make sure pipe ``fd`` can hold ``nbytes`` unread bytes; returns its capacity.

    Linux ``F_GETPIPE_SZ``, raised with ``F_SETPIPE_SZ`` if smaller; a clear
    ``OSError`` if it cannot grow. Returns None where it cannot be queried
    (not Linux; MAX_PREFETCH records are 16 KiB, below any default pipe)."""
    if _F_GETPIPE_SZ is None:
        return None
    have = fcntl.fcntl(fd, _F_GETPIPE_SZ)
    if have >= nbytes:
        return have
    try:
        return fcntl.fcntl(fd, _F_SETPIPE_SZ, nbytes)
    except OSError as e:
        raise OSError(e.errno, f"the reader task pipe holds {have} B, the {nbytes} B of "
                      f"tasks in flight need more and it cannot grow ({e.strerror}; "
                      "/proc/sys/fs/pipe-max-size, pipe-user-pages-soft): lower "
                      "--prefetch") from None


def _check_fd_room(n_slots):
    """Refuse up front when RLIMIT_NOFILE cannot hold the ring's fds (Python < 3.13).

    Before 3.13 every ring slot's mapping pins one fd in the run process and in
    each reader. A clear message here beats an EMFILE half-way through the
    ring; ``n_slots`` plus FD_MARGIN for everything else must fit."""
    if _MMAP_KW:
        return
    soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    need = n_slots + FD_MARGIN
    if soft != resource.RLIM_INFINITY and soft < need:
        raise OSError(errno.EMFILE,
                      f"the block ring's {n_slots} slots keep one file descriptor each "
                      f"per process before Python 3.13; RLIMIT_NOFILE is {soft}, need "
                      f">= {need}: raise it (ulimit -n) or lower --prefetch")


FD_MARGIN = 256  # fds for everything but the ring (pipes, sockets, files, libraries)


class _ProcReaders:
    """``n`` spawned reader processes, one task pipe they all pull from, one reply
    pipe each, and the block ring.

    ``submit`` writes a 16-byte ``(tid, slot)`` record straight into the shared pipe
    (no feeder thread, so nothing in the parent must run later, e.g. while a
    GIL-holding engine call blocks it); whichever reader is free takes it, so no file
    waits behind a slow one while a reader idles. Records are below ``PIPE_BUF``, so
    writes are atomic and readers need no lock. (A ``multiprocessing`` queue would not
    do: an idle reader holds its lock, so terminating one wedges the others.)

    The parent assigns ring slots with the task and frees a slot only after its reply
    is consumed and the block dropped, or after ``abort`` killed every reader that
    could still write it, so no two writers ever share a slot. ``abort`` kills all
    readers if anything is in flight; only ``start`` (untimed) respawns them, and a
    pass that finds a reader dead raises ``ReaderDied``."""

    def __init__(self, n, fs, read_fn, cpu_threads, io_threads, slot_bytes, n_slots,
                 files, kw, pin=None):
        self.n, self.fs, self.read_fn = n, fs, read_fn
        self.pin = pin  # --h2d pinned: _BlockRing registers its slots with it
        self.cpu_threads, self.io_threads = cpu_threads, io_threads
        self.slot_bytes, self.n_slots = slot_bytes, n_slots
        self.max_inflight = max(1, n_slots - 1)  # tasks in flight at most (prefetch)
        self.files, self.kw = files, kw  # tid -> (path, size, footer); read settings
        self.ctx = mp.get_context("spawn")
        self.procs = [None] * n   # (Process, Connection) per reader
        self.running = [None] * n  # per reader: the (tid, path) it said it started
        self.outstanding = {}     # tid -> path: submitted, no reply consumed yet
        self.chan = [None, None]  # the shared task pipe: (read end, write end)
        self._aborting = False
        self.prefix = f"{SHM_PREFIX}{os.getpid()}-{secrets.token_hex(4)}-"
        self.ring = None
        self.pa_threads = [(None, None)] * n  # per reader: what it reports (cpu, io)
        self.filter_setup = [None] * n  # per reader: seconds building its DuckFilter
        self._fin = weakref.finalize(self, _ProcReaders._shutdown, self.procs, self.chan,
                                     self.prefix)

    @property
    def busy(self):
        """reader -> the (tid, path) it is reading, for readers with a file."""
        return {i: t for i, t in enumerate(self.running) if t is not None}

    def start(self):
        """Create the ring and task pipe if needed, spawn missing readers, wait for each ``ready``.

        The ring is made on the first call, the pipe on the first call or after
        ``abort``. ``ready`` means the reader has imported its modules, sized
        its pools, warmed up and mapped the ring. Untimed when called before the
        passes (``ParquetSource.start``). Anything still in flight (a pass left
        open, or an abort cut short) is aborted first, so the pipe is empty and
        no old reader can still write a slot when a dead reader is replaced."""
        self.abort()  # no-op when idle
        if self.ring is None:
            self.ring = _BlockRing(self.prefix, self.slot_bytes, self.n_slots, self.pin)
        if self.chan[0] is None:
            self.chan[:] = self.ctx.Pipe(duplex=False)
            # every in-flight task fits in the pipe at once, so submit's
            # blocking os.write never waits on a reader (see MAX_PREFETCH)
            ensure_pipe_room(self.chan[1].fileno(), self.max_inflight * _TASK.size)
        new = []
        with _main_not_reimported():
            for i, slot in enumerate(self.procs):
                if slot is not None and slot[0].is_alive():
                    continue
                if slot is not None:
                    slot[1].close()
                parent, child = self.ctx.Pipe(duplex=True)
                p = self.ctx.Process(target=_reader_main, name=f"pq-reader-{i}", daemon=True,
                                     args=(self.chan[0], child, self.fs, self.read_fn,
                                           self.cpu_threads[i], self.io_threads[i], i,
                                           self.n, self.prefix, len(self.ring), self.files,
                                           self.kw))
                p.start()
                child.close()
                self.procs[i] = (p, parent)
                self.running[i] = None
                new.append(i)
        for i in new:
            p, conn = self.procs[i]
            ready = mpc.wait([conn, p.sentinel])
            if conn not in ready and not conn.poll():
                raise ReaderDied(f"reader process {p.pid} exited during start-up "
                                 f"(exit code {p.exitcode})")
            msg = conn.recv()
            assert msg[0] == "ready", msg
            self.pa_threads[i] = (msg[2], msg[3])
            self.filter_setup[i] = msg[4] if len(msg) > 4 else None

    def check_ready(self):
        """At the start of a pass: check every reader is up and idle, nothing in flight.

        Raises ``ReaderDied`` naming the reader otherwise. A pass never respawns
        one (spawn + imports + warm-up would land in its timed span); only
        ``start()``, untimed, does."""
        if self.outstanding or self._aborting or any(t is not None for t in self.running):
            raise ReaderDied(f"{len(self.outstanding)} file(s) still in flight from an earlier "
                             "pass (left open, or its abort was cut short): call start() "
                             "before the next pass")
        for r, e in enumerate(self.procs):
            if e is None:
                raise ReaderDied(f"reader {r} is not running (an earlier pass was aborted, "
                                 "which stops the readers): call start() before the next pass")
            p = e[0]
            if not p.is_alive():
                p.join(1)
                raise ReaderDied(f"reader process {p.pid} (reader {r}) died (exit code "
                                 f"{p.exitcode}) while idle between passes; start() "
                                 "respawns it, untimed")

    def submit(self, tid, path, slot):
        """Put task ``tid`` (``files[tid]``, read into ring ``slot``) on the shared task pipe.

        The first free reader takes it. A reader found dead here (it died while
        idle) raises ``ReaderDied`` naming it and ``path``. The write never
        blocks: ``start`` sized the pipe for every task that can be in flight
        (``max_inflight``), and more are refused here, since a blocked write with
        no reply being read would deadlock."""
        if len(self.outstanding) >= self.max_inflight:
            raise RuntimeError(f"{len(self.outstanding)} tasks already in flight, the "
                               f"task pipe was sized for {self.max_inflight}")
        for r, (p, _) in enumerate(self.procs):
            if not p.is_alive():
                p.join(1)
                was = f", while reading {self.running[r][1]}" if self.running[r] else ""
                raise ReaderDied(f"reader process {p.pid} (reader {r}) is gone (exit code "
                                 f"{p.exitcode}{was}); found while handing out {path}")
        os.write(self.chan[1].fileno(), _TASK.pack(tid, slot))
        self.outstanding[tid] = path

    def collect(self, results, timeout=None):
        """Wait for reader messages; store replies in ``results`` (tid -> message).

        Blocks until a reader says something (``timeout`` 0 never blocks: the
        --h2d-overlap poll), and notes which reader started which file. A reader
        that died raises ``ReaderDied`` at once, naming the file it was on,
        whichever file the consumer is waiting on (any reader may hold any queued
        file, so a death while idle is fatal too)."""
        waitables = {}
        for r, (p, conn) in enumerate(self.procs):
            waitables[conn] = r
            waitables[p.sentinel] = r
        for r in sorted({waitables[obj] for obj in mpc.wait(list(waitables), timeout)}):
            p, conn = self.procs[r]
            try:
                while conn.poll():
                    msg = conn.recv()
                    if msg[0] == "start":
                        self.running[r] = (msg[1], self.outstanding[msg[1]])
                        continue
                    self.running[r] = None
                    self.outstanding.pop(msg[1])
                    results[msg[1]] = msg
            except (EOFError, OSError):
                pass
            if not p.is_alive():
                p.join(1)
                if self.running[r] is not None:
                    raise ReaderDied(f"reader process {p.pid} died (exit code {p.exitcode}) "
                                     f"while reading {self.running[r][1]}")
                raise ReaderDied(f"reader process {p.pid} (reader {r}) died (exit code "
                                 f"{p.exitcode}) while idle or just starting a task (no "
                                 f"file named yet), {len(self.outstanding)} file(s) in flight")

    def abort(self):
        """Fast close: terminate every reader if anything is in flight.

        Applies when a task is in flight or an earlier abort was cut short:
        terminate every reader (a multi-GB S3 read is not waited for, and an
        idle one could pick up a record still in the pipe), reap them and drop
        the task pipe. With nothing in flight the readers stay up. The ring
        stays (the caller gives the aborted tasks' slots back). Re-entrant: the
        bookkeeping is cleared first and each reader is nulled only once
        reaped, so an abort interrupted half-way (Ctrl-C) is finished by the
        next one (``close`` calls it again)."""
        if self.outstanding or any(t is not None for t in self.running):
            self._aborting = True
        if not self._aborting:
            return
        self.outstanding.clear()
        self.running[:] = [None] * self.n
        live = [(r, e) for r, e in enumerate(self.procs) if e is not None]
        for _, (p, _) in live:
            p.terminate()
        for r, (p, conn) in live:
            p.join(5)
            if p.is_alive():
                p.kill()
                p.join(5)
            conn.close()
            self.procs[r] = None
        for i, c in enumerate(self.chan):
            if c is not None:
                c.close()
                self.chan[i] = None
        self._aborting = False

    def close(self):
        """Abort, stop the readers and unlink the ring."""
        self.abort()  # never wait for an in-flight read (no-op when idle)
        self._fin()
        if self.ring is not None:
            self.ring.close()

    @staticmethod
    def _shutdown(procs, chan, prefix):
        """Finalizer: send each reader a stop record, reap them, close the pipe, remove the ring."""
        live = [e for e in procs if e is not None]
        if chan[1] is not None:
            for _ in live:  # one stop record per reader
                try:
                    os.write(chan[1].fileno(), _TASK.pack(-1, -1))
                except (OSError, ValueError):
                    break
        for i, slot in enumerate(procs):
            if slot is None:
                continue
            p, conn = slot
            p.join(2)
            if p.is_alive():
                p.terminate()
                p.join(5)
            conn.close()
            procs[i] = None
        for i, c in enumerate(chan):
            if c is not None:
                c.close()
                chan[i] = None
        _sweep_shm(prefix)


def _block_itemsize(footer, dtype):
    """Bytes per value of the block ``read_file`` makes from a file with
    ``footer``: the host ``dtype`` if one is asked for, else the stored float
    width (anything else is widened to float32, see ``table_vectors``)."""
    if dtype is not None:
        return np.dtype(dtype).itemsize
    return {16: 2, 32: 4}.get(footer.value_bits, 4)


class ParquetSource:
    """Stream Parquet files in order using parallel reader processes.

    Up to ``prefetch`` reads run concurrently across ``workers`` processes.
    Readers start once and persist across passes, decoding directly into
    reusable shared-memory buffers. Blocks are returned in file order without
    copying, and buffers are reused only after the consumer releases them.

    ``prefetch=0`` reads files inline. Each pass resets read statistics.
    Reader failures stop the pipeline, and early termination requires closing
    the iterator.

    Optional filtering evaluates predicates during reading and can compact
    matching rows while preserving their original file offsets.
    """

    def __init__(self, uri, column="embedding", prefetch=2, workers=None,
                 dtype=None, fs=None, files=None, read_mode="native",
                 range_concurrency=NATIVE_CONCURRENCY, range_scope="total",
                 range_bytes=NATIVE_PART_BYTES, pa_cpu_threads=0,
                 pa_io_threads=0, read_fn=None, reader_pool="split",
                 codec=None, pin=None, extra_slots=0,
                 filter_job=None, filter_compact=None):
        if not 0 <= prefetch <= MAX_PREFETCH:
            raise ValueError(f"prefetch must be in [0, {MAX_PREFETCH}] (MAX_PREFETCH: "
                             "the reader task pipe and the block ring are sized for it)")
        if read_mode not in FILE_MODES:
            raise ValueError(f"read mode {read_mode!r} not in {FILE_MODES}")
        if reader_pool not in READER_POOLS:
            raise ValueError(f"reader pool {reader_pool!r} not in {READER_POOLS}")
        if fs is None or files is None:
            fs, files = list_parquet(uri)
        self.fs, self.files = fs, files
        self.column, self.dtype = column, dtype
        self.prefetch = prefetch
        self.workers = workers or default_workers()
        self.read_mode = read_mode
        self.read_mode_effective, self.column_share = resolve_read_mode(
            fs, files[0], column, read_mode)
        self.range_bytes = range_bytes
        # no more readers (and ring slots, see start) than files to read
        self.readers = min(prefetch, self.workers, len(files)) if prefetch else 1
        self.range_concurrency = range_concurrency_for(self.readers, range_concurrency,
                                                       range_scope)
        # what actually reads: reader processes, or inline (prefetch 0)
        self.reader = "process" if prefetch else "inline"
        if self.read_mode_effective == "native":
            native_import()  # fail now, not in the first read
        # sparse / multivector: blocks are packed byte arrays decoded, counted and sized
        # by the codec
        self.codec = codec
        if codec is not None and read_fn is None:
            read_fn = functools.partial(read_file, codec=codec)
        self.read_fn = read_fn
        # n/a for the inline reader; with one reader full == split, so record it as split
        self.reader_pool = ("split" if self.readers == 1 else reader_pool) \
            if self.reader == "process" else "n/a"
        self.reader_pa_threads = reader_pool_threads(pa_cpu_threads, pa_io_threads,
                                                     self.readers, reader_pool)
        if self.reader == "process" and reader_pool == "split" \
                and 0 < pa_cpu_threads < self.readers:
            warnings.warn(f"--cpu-thread-count {pa_cpu_threads} < {self.readers} reader "
                          f"processes: each gets 1 decode thread, {self.readers} in total",
                          stacklevel=2)
        # --h2d pinned page-locks the ring slots once; the inline reader has no ring
        if pin is not None and self.reader != "process":
            raise ValueError("pinned ring slots need reader processes (--prefetch > 0)")
        self.pin = pin
        # --h2d-overlap: the consumer holds two blocks (the one searched and
        # the next, uploading), so the ring gets one more slot and never grows
        self.extra_slots = int(extra_slots)
        # filtered search: readers also read and evaluate the filter columns; masks
        # travel in the block's slot
        self.filter_job = filter_job
        if filter_job is not None and codec is not None:
            raise ValueError("filtering a sparse corpus is not supported")
        # readers compact each block after filtering it; None = hand blocks over whole
        if filter_compact is not None and filter_job is None:
            raise ValueError("filter_compact needs a filter_job")
        self.filter_compact = filter_compact
        self._flt_pool = None                # the inline reader's DuckFilter (start())
        self._flt_lock = threading.Lock()
        self._mask_layout = None             # (offset, capacity) within a slot
        self._keep_layout = None             # (offset, capacity): compaction's keep rows
        self._procs = None
        self.footers = {}  # arrow mode: path -> Footer, read once per file
        self.stats = Stages()
        self.footer_read_s = None  # start()'s footer read, seconds

    # -- process readers: lifecycle
    def read_footers(self):
        """Read every file's footer into ``footers`` (one small read per uncached file); returns it."""
        for ref in self.files:
            if ref.path not in self.footers:
                if ref.size == 0:
                    raise ValueError(f"{ref.path}: empty file")
                self.footers[ref.path] = read_footer(self.fs, ref.path, self.column)
        return self.footers

    def max_block_bytes(self):
        """Bytes of the largest block any file decodes to (rows x d x itemsize, from the footers).

        Footers not cached yet are read here. A file whose footer gives no
        dimension (not a plain list / fixed_size_list of floats) is skipped:
        its read fails on its own."""
        out, footers = 0, self.read_footers()
        for ref in self.files:
            foot = footers[ref.path]
            if self.codec is not None:
                out = max(out, self.codec.block_bytes(foot, self.dtype) or 0)
            elif foot.dim:
                out = max(out, foot.num_rows * foot.dim * _block_itemsize(foot, self.dtype))
        return max(1, out)

    def start(self):
        """Untimed set-up before the passes: read every footer (so no pass fetches one),
        then bring up the reader processes and the ring.

        ``footer_read_s`` is that footer read's wall time. It is reported in its own
        column, never added to the timed seconds."""
        t_foot = now()
        self.read_footers()
        self.footer_read_s = now() - t_foot
        if self.reader != "process":
            self._build_filter_pool()
            if self.read_mode_effective == "native" and self.files:
                native_warm_up(self.fs, self.files[0].path, self.range_concurrency,
                               self.range_bytes)
            return self
        if self._procs is None:
            _check_fd_room(min(self.prefetch, len(self.files)) + 1 + self.extra_slots)
            kw = dict(column=self.column, dtype=self.dtype, mode=self.read_mode_effective,
                      range_concurrency=self.range_concurrency, range_bytes=self.range_bytes)
            slot_bytes = self.max_block_bytes()
            if self.filter_job is not None:
                from bench.data.filters import reader as filtered
                max_rows = max(self.footers[r.path].num_rows for r in self.files)
                offset = filtered.aligned(slot_bytes)
                n_masks = self.filter_job.max_masks()
                cap = filtered.mask_capacity(n_masks, max_rows)
                self._mask_layout = (offset, cap)
                slot_bytes = offset + cap
                kw.update(filter_job=self.filter_job, mask_offset=offset, mask_capacity=cap)
                if self.filter_compact is not None:
                    # after the masks: the kept rows' file row ids (int64),
                    # with room for every row of the largest file
                    k_off, k_cap = filtered.aligned(slot_bytes), max(1, max_rows * 8)
                    self._keep_layout = (k_off, k_cap)
                    slot_bytes = k_off + k_cap
                    kw.update(filter_compact=self.filter_compact, keep_offset=k_off,
                              keep_capacity=k_cap)
            n_slots = min(self.prefetch, len(self.files)) + 1 + self.extra_slots
            if self.filter_job is not None:
                # slots have room for every possible mask: refuse a ring that
                # would not fit in memory, rather than swap or fail mid-run
                avail = filtered.mem_available()
                if avail is not None and n_slots * slot_bytes > filtered.MASK_RING_MEM_FRAC * avail:
                    raise RuntimeError(
                        f"filtered ring: {n_slots} slots x {slot_bytes / 2**20:.0f} MiB "
                        f"(room for all {n_masks} possible masks of a {max_rows}-row file) "
                        f"> {filtered.MASK_RING_MEM_FRAC:.0%} of MemAvailable "
                        f"({avail / 2**30:.1f} GiB): use fewer readers (--prefetch)")
            self._procs = _ProcReaders(
                self.readers, self.fs, self.read_fn, *self.reader_pa_threads,
                slot_bytes=slot_bytes,
                n_slots=n_slots,
                files=[(r.path, r.size, self.footers.get(r.path)) for r in self.files], kw=kw,
                pin=self.pin)
        self._procs.start()
        return self

    def _build_filter_pool(self):
        """Build the inline reader's one DuckFilter (own DuckDB connection), before the passes.

        Query prep and the self-checks then never fall inside a timed pass. The
        reader checks it out per file (``_read``)."""
        if self.filter_job is None or self._flt_pool is not None:
            return
        import queue
        with self._flt_lock:  # a pass without start(): several threads get here
            if self._flt_pool is not None:
                return
            pool = queue.SimpleQueue()
            t = now()
            pool.put(self.filter_job.build())
            self._flt_setup = now() - t
            self._flt_pool = pool

    _flt_setup = None

    @property
    def filter_setup_s(self):
        """Wall seconds the readers spent building their DuckFilter engines, at start-up.

        Covers query-side prep and, once per process, the Unicode self-checks,
        all outside every timed pass. Process readers build in parallel, so
        this is the slowest one. None: no filter, or not built yet."""
        if self._procs is not None:
            got = [v for v in self._procs.filter_setup if v is not None]
            return max(got) if got else None
        return self._flt_setup

    def close(self):
        """Stop the reader processes and unlink the ring (no-op for the inline reader)."""
        if self._procs is not None:
            self._procs.close()
            self._procs = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def reader_pa_threads_effective(self):
        """(cpu, io) Arrow pool sizes the reader processes report, each a list
        by reader (``pool_split_str`` makes the CSV form); (None, None) before
        they started, or for the inline reader."""
        if self._procs is None or any(c is None for c, _ in self._procs.pa_threads):
            return None, None
        cpu, io = zip(*self._procs.pa_threads)
        return list(cpu), list(io)

    @property
    def shm_prefix(self):
        """Name prefix of this source's shared-memory segments, or None."""
        return self._procs.prefix if self._procs is not None else None

    @property
    def ring(self):
        """The shared-memory ring (process readers, once started) or None."""
        return self._procs.ring if self._procs is not None else None

    def _rows(self, x):
        return x.shape[0] if self.codec is None else self.codec.rows(x)

    def _read(self, ref):
        """Inline read of one file (plus filter / compaction); returns ``(x, rec)``."""
        rec = {}
        fn = self.read_fn or read_file  # looked up per call (tests patch it)
        kw = {}
        if self.filter_job is not None:
            kw.update(extra_columns=self.filter_job.fields, extra={})
        x = fn(self.fs, ref.path, self.column, self.dtype, self.stats,
               ref.size, self.read_mode_effective, self.range_concurrency,
               self.range_bytes, self.footers, rec=rec, **kw)
        if self.filter_job is not None:
            from bench.data.filters import reader as filtered
            self._build_filter_pool()  # no-op after start(); else built once, here
            table = kw["extra"].pop("table")
            if table.num_rows != self._rows(x):
                raise RuntimeError(f"{ref.path}: {table.num_rows} filter rows vs "
                                   f"{self._rows(x)} vectors")
            flt = self._flt_pool.get()
            try:
                rec["filter_start"] = now()
                masks, info = filtered.evaluate(flt, table, self.stats, rec)
            finally:
                self._flt_pool.put(flt)
            rec["filter_publish"] = 0.0
            if self.filter_compact is not None:
                # in place in this read's own array (a fresh copy if read-only)
                x, masks, info, rec["compact"] = filtered.compact_block(
                    self.filter_compact, x, masks, info, self.stats, rec)
            # after compaction, as the process readers stamp it: the trace's
            # filter window is eval + compaction
            rec["filter_end"] = now()
            filtered.finish_timing(self.stats, rec)
            rec["filter_block"] = filtered.block_from(info, masks)
        return x, rec

    def __iter__(self):
        self.stats = Stages()
        if self.reader == "process":
            return self._iter_processes()
        return self._iter_inline()

    def stream(self, n_files=None, pollable=False):
        """One pass like ``iter()``, over the first ``n_files`` files only.

        ``n_files`` None / 0 = all (used for --warmup-files). With ``pollable``
        the iterator also has ``poll()`` (``_Pollable``: the next block if
        already read, ``NOT_READY`` if not, never blocking); reader processes
        only, used for --h2d-overlap."""
        self.stats = Stages()
        files = self.files[:n_files] if n_files else self.files
        if not pollable:
            return self._iter_processes(files) if self.reader == "process" \
                else self._iter_inline(files)
        if self.reader != "process":
            raise ValueError("a pollable stream needs reader processes")
        return _Pollable(lambda ctl: self._iter_processes(files, ctl))

    def _iter_inline(self, files=None):
        files = self.files if files is None else files
        offset = 0
        for ref in files:
            x, rec = self._read(ref)
            comp = rec.pop("compact", None)
            n = self._rows(x) if comp is None else comp.n_rows  # the file's rows
            out = [Block(offset, x, ref.path, rec, rec.pop("filter_block", None), comp)]
            del comp
            del x  # the consumer's block only: see _iter_processes
            yield out.pop()
            offset += n

    def _iter_processes(self, files=None, ctl=None):
        files = self.files if files is None else files
        if self._procs is None:
            self.start()  # never started: the first pass brings the readers up
        procs, ring = self._procs, self._procs.ring
        # a reader that died between passes, or readers an abort stopped, are
        # not respawned here inside the timed pass (ReaderDied); start()
        # respawns them, untimed
        procs.check_ready()
        offset = 0
        todo = enumerate(files)  # a prefix of self.files: tid = index in both
        inflight = deque()   # (tid, ref, slot) in file order, at most `prefetch`
        results = {}

        def add(item):
            # take a free ring slot for the block and put the task on the shared
            # task pipe right away, so the first free reader starts it
            tid, ref = item
            inflight.append((tid, ref, ring.take()))
            procs.submit(tid, ref.path, inflight[-1][2])

        try:
            for item in todo:
                add(item)
                if len(inflight) == self.prefetch:
                    break
            while inflight:
                tid, ref, slot = inflight[0]
                while tid not in results:
                    if ctl is not None and ctl.nonblock:
                        # _Pollable.poll(): look, never wait; NOT_READY hands
                        # control back with nothing consumed (the next call,
                        # poll or next, resumes right here)
                        procs.collect(results, 0)
                        if tid not in results:
                            yield NOT_READY
                            continue
                    else:
                        procs.collect(results)
                msg = results.pop(tid)
                inflight.popleft()
                if msg[0] == "err":
                    ring.put(slot)
                    raise msg[2] from _RemoteTraceback(msg[3])
                _, _, shape, dtype, wstats, rec, footer = msg
                t0 = now()
                # the slot is the array's now, free again once it is dropped
                fblock = comp = None
                info = rec.pop("filter_info", None)
                if info is not None:
                    from bench.data.filters import reader as filtered
                    # one base over the slot: the vectors (compacted by the
                    # reader under gather / union), the masks unless they
                    # overflowed into the reply, the compaction's keep rows
                    parts = [(0, shape, np.dtype(dtype))]
                    inline = info.pop("masks_inline", None)
                    if inline is None:
                        parts.append((self._mask_layout[0], (info["m"], info["nb"]), np.uint8))
                    cinfo = info.pop("compact", None)
                    if cinfo is not None:
                        parts.append((self._keep_layout[0], (cinfo[1],), np.int64))
                    got = ring.view_parts(slot, parts)
                    x = got[0]
                    fblock = filtered.block_from(info, inline if inline is not None else got[1])
                    if cinfo is not None:
                        comp = filtered.Compacted(cinfo[0], got[-1], cinfo[2], cinfo[3])
                    del got, inline
                else:
                    x = ring.view(slot, shape, np.dtype(dtype))
                rec["handoff"] = now() - t0
                _merge_stats(self.stats, wstats)
                for key, col in (("publish", "publish"), ("handoff", "handoff"),
                                 ("cpu_user", "worker_cpu_user"),
                                 ("cpu_sys", "worker_cpu_sys")):
                    self.stats.add(col, rec[key])
                if footer is not None:
                    self.footers[ref.path] = footer
                # submit only now that the head is read: `prefetch` reads in
                # flight at most, the head included
                nxt = next(todo, None)
                if nxt is not None:
                    add(nxt)
                n = self._rows(x) if comp is None else comp.n_rows  # the file's rows
                # yield a Block this frame holds no reference to, so the consumer's
                # ``del`` frees the slot in its own loop, not in the next ``next()``
                out = [Block(offset, x, ref.path, rec, fblock, comp)]
                del x, fblock, comp
                yield out.pop()
                offset += n
        except BaseException:
            # consumer-side error / close (GeneratorExit), a failed read, a dead
            # reader, Ctrl-C: don't wait for in-flight reads; kill all readers
            # (any may hold a queued task) and give the unconsumed tasks'
            # slots back
            try:
                procs.abort()
            finally:
                # give slots back even if the abort itself is interrupted
                # (Ctrl-C): any reader it left alive is killed by the next
                # start() / close() (abort is re-entrant) before a slot can be
                # handed out again
                for _, _, slot in inflight:
                    ring.put(slot)
            raise


class _NotReady:
    """Type of the NOT_READY sentinel."""

    def __repr__(self):
        return "NOT_READY"


# _Pollable.poll(): the next block has not been read yet
NOT_READY = _NotReady()


class _Pollable:
    """A pass over a ParquetSource (``stream(pollable=True)``) that can also be polled.

    ``poll()`` returns the next block if its reader has already replied (handed
    over exactly as ``next()`` would: the view, ``handoff``, the next task
    submitted), ``NOT_READY`` if not, and None at the end; it never blocks.
    ``next()`` blocks as usual. ``close()`` aborts like closing the generator.
    --h2d-overlap polls for block i+1 right before launching search i."""

    def __init__(self, make):
        self._ctl = types.SimpleNamespace(nonblock=False)
        self._gen = make(self._ctl)
        self._done = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        self._ctl.nonblock = False
        try:
            x = next(self._gen)
        except StopIteration:
            self._done = True
            raise
        assert x is not NOT_READY
        return x

    def poll(self):
        """The next block if already read, ``NOT_READY`` if not, None at the end."""
        if self._done:
            return None
        self._ctl.nonblock = True
        try:
            return next(self._gen)
        except StopIteration:
            self._done = True
            return None
        finally:
            self._ctl.nonblock = False

    def close(self):
        self._gen.close()


def rss_bytes(field="VmRSS", path="/proc/self/status"):
    """This process's ``VmRSS`` (or ``VmHWM``, its high-water mark) in bytes;
    None where /proc is missing."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


class InMemorySource:
    """The corpus decoded into host RAM, untimed: the ``--verify`` oracle's reload.

    ``load`` decodes with ``arrow``, independent of the native timed reads. Each file
    goes straight into its own page-aligned array in record batches, and Arrow's pool
    returns freed memory after every file, so the loaded corpus costs about 1x its
    blocks instead of growing with each read's transient."""

    def __init__(self, blocks):
        self.blocks = blocks
        self.stats = Stages()

    @classmethod
    def load(cls, uri, column="embedding", dtype=None, read_mode=ARROW, fs=None, files=None,
             batch_bytes=PRELOAD_BATCH_BYTES, codec=None):
        """Read every file (arrow by default) into a list of Blocks with running row offsets."""
        if fs is None or files is None:
            fs, files = list_parquet(uri)
        mode, _ = resolve_read_mode(fs, files[0], column, read_mode)
        blocks, offset = [], 0
        pool = pa.default_memory_pool()
        pool.release_unused()  # before a reload
        for ref in files:
            x = read_file(fs, ref.path, column, dtype, None, ref.size, mode,
                          batch_bytes=batch_bytes, codec=codec)
            blocks.append(Block(offset, x, ref.path))
            offset += x.shape[0] if codec is None else codec.rows(x)
            pool.release_unused()  # this file's transient, before the next one's
        return cls(blocks)

    # iterable of preloaded blocks for run_pass (used by CPU tests)
    def __iter__(self):
        return iter(self.blocks)

    def stream(self, n_files=None, pollable=False):
        """``iter()`` over the first ``n_files`` blocks (None / 0 = all); never
        pollable."""
        if pollable:
            raise ValueError("a pollable stream needs a ParquetSource with reader processes")
        return iter(self.blocks[:n_files] if n_files else self.blocks)
