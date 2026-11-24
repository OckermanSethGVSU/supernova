"""An experiment of experiments/ planned into a run dir for the check steps."""

import json
import os

from bench import experiment as E
from bench import runner


CB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(CB, "experiments")
DUMMY = {"sparse_fineweb": ["corpus.path=s3://DUMMY-fineweb/corpus/", "queries.nq=1000"],
         # dummy values for experiments whose data is not set yet
         "multivector_pubmed": ["corpus.path=s3://DUMMY-pubmed/corpus/", "queries.nq=1000",
                                "queries.path=~/data/queries_pubmed.parquet"],
         "filtered_fineweb_smoke": ["io.files_in_flight=4", "io.gets=512", "io.part_mib=16"]}


def _load(name, tmp_path):
    path = os.path.join(CB, E.path_of(name))
    raw = runner._set(E.load_raw(path), DUMMY.get(name, []))
    exp = E.finish(raw, path)
    d = runner.new_run_dir(str(tmp_path / "res"))
    open(os.path.join(d, "experiment.json"), "w").write(E.dumps(exp))
    return json.load(open(os.path.join(d, "experiment.json"))), d
