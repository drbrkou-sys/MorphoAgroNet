from pathlib import Path as _ProjectPath
import sys as _project_sys
_project_sys.path.insert(0,str(_ProjectPath(__file__).resolve().parents[1]))
import argparse,json
from pathlib import Path
import numpy as np
from morphoagronet.evaluation.statistics import compare_runs,holm_adjust,stratified_bootstrap_delta,mcnemar
p=argparse.ArgumentParser();p.add_argument('--model-runs',required=True,help='JSON list of macro-F1 values')
p.add_argument('--baseline-runs',required=True);p.add_argument('--predictions-a');p.add_argument('--predictions-b')
p.add_argument('--output',default='results/statistics.json');a=p.parse_args()
r=compare_runs(json.loads(Path(a.model_runs).read_text()),json.loads(Path(a.baseline_runs).read_text()))
if a.predictions_a and a.predictions_b:
    A=json.loads(Path(a.predictions_a).read_text());B=json.loads(Path(a.predictions_b).read_text())
    if A['true']!=B['true']:raise ValueError('Paired predictions have different sample order/labels')
    r['bootstrap']=stratified_bootstrap_delta(A['true'],A['pred'],B['pred'])
    r['mcnemar']=mcnemar(A['true'],A['pred'],B['pred'])
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
