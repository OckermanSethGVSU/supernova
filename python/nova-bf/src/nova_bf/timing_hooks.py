"""Optional timing hooks for `run_compute` (observe only).

A benchmark harness can pass an object with these four methods as
`run_compute(..., timing_hooks=obj)` to time the search with its own clock.
nova-bf calls them at fixed points and does nothing else with them:

    corpus_start()  once, just before the scan's corpus reader threads (and
                    the `tiebreak='id'` id-column pass) start: no corpus data
                    has been read yet. Only file listing and footer/schema
                    reads happen before it. With a multivector search it comes
                    earlier, just before `_sample_mean_doc_tokens` (which reads
                    the first file's token column); then `pause()` follows
                    that sample and the scan position calls `resume()`.
    pause()         once per search, right after that search's final top-K
                    sort and copy to the host, before its result is decoded
                    (point ids) and its output write is submitted.
    resume()        once per search, after that decode and write submission.
    done()          once, after the last search's sort/decode loop, before
                    waiting for the outstanding output writes.

So a run with S searches calls corpus_start, S x (pause, resume), done, in
that order (a multivector run: one extra pause, resume right after corpus_start). They are never called per file or per slice, and no GPU sync is
added for them: the final sort's `.cpu()` copies already wait for the device.

An exception in run_compute can leave the sequence incomplete (no `done()`).
nova-bf does not call anything on the error path, so a caller must treat a
missing `done()` as a failed run. A hook that raises fails the run.

`NoHooks` (the default, `timing_hooks=None`) does nothing.
"""

from __future__ import annotations


class NoHooks:
    """The default: every hook is a no-op."""

    __slots__ = ()

    def corpus_start(self) -> None:
        pass

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass

    def done(self) -> None:
        pass


NO_HOOKS = NoHooks()
