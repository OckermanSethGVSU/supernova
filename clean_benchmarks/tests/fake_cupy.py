"""A numpy-backed stand-in for the CuPy calls cuVS's own recipe makes (cuvs_engine.CalcTruthMerge,
NVIDIA's calc_truth): asarray / concatenate / argsort / take_along_axis / asnumpy, and the
cuda.ExternalStream / cuda.set_allocator the harness wires. Every call is logged in ``log``
(name, shapes), so tests can check the recipe's op sequence. ``argsort`` is numpy's default
(as CuPy's default), so ties follow the library, not an id rule."""

import contextlib
import sys
import types

import numpy as np


def make(log=None):
    log = [] if log is None else log
    cp = types.ModuleType("cupy")
    cp.log = log

    def shp(a):
        return tuple(np.shape(a))

    def asarray(a, dtype=None):
        out = np.asarray(a, dtype=dtype)
        log.append(("asarray", shp(out), str(out.dtype)))
        return out

    def concatenate(arrs, axis=0):
        out = np.concatenate(arrs, axis=axis)
        log.append(("concatenate", tuple(shp(a) for a in arrs), axis))
        return out

    def argsort(a, axis=-1):
        log.append(("argsort", shp(a), axis))
        return np.argsort(a, axis=axis)

    def take_along_axis(a, idx, axis):
        log.append(("take_along_axis", shp(a), shp(idx), axis))
        return np.take_along_axis(a, idx, axis=axis)

    def asnumpy(a):
        log.append(("asnumpy", shp(a)))
        return np.array(a)

    cp.asarray, cp.concatenate, cp.argsort = asarray, concatenate, argsort
    cp.take_along_axis, cp.asnumpy = take_along_axis, asnumpy
    cuda = types.ModuleType("cupy.cuda")
    cuda.streams = []

    def ExternalStream(ptr):
        cuda.streams.append(ptr)
        return contextlib.nullcontext()

    cuda.ExternalStream = ExternalStream
    cuda.allocators = []
    cuda.set_allocator = lambda a: cuda.allocators.append(a)
    cp.cuda = cuda
    return cp


def install(monkeypatch, log=None):
    cp = make(log)
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setitem(sys.modules, "cupy.cuda", cp.cuda)
    return cp
