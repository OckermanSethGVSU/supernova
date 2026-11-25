Experiment files for the tests. The tests read only these, never experiments/ (which changes
as settings are tuned).

- Top level: small experiments on the generated nova-bf fixture data (tests/fixtures/novabf,
  made by its make_fixtures.py), runnable on a CPU with `bench run --dev`. Data paths are
  relative to the clean_benchmarks root. The nb_*.yaml nova-bf configs mirror
  experiments/novabf/*.yaml (same ${NOVABF_*} variables, params verbatim) except k and the CPU
  multivector kernel.
- dense/ sparse/ multivector/ filtered/ novabf/: copies of experiments/ at one point in time,
  for tests that need an experiment of the full shape (they are never run, only planned).
- sweeps/: a few sweep files on those copies.
