from pathlib import Path as _ProjectPath
import sys as _project_sys
_project_sys.path.insert(0,str(_ProjectPath(__file__).resolve().parents[1]))
import argparse,json
from pathlib import Path
from morphoagronet.training.trainer import train_experiment
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',default='results/ablations')
p.add_argument('--seeds',nargs='+',type=int,default=[42,123,2024,2025,3407]);p.add_argument('--epochs',type=int,default=100)
p.add_argument('--batch',type=int,default=32);p.add_argument('--no-pretrained',action='store_true')
a=p.parse_args();variants={
 'full':{},'without_mlge':{'ablations':{'use_mlge':False}},
 'without_ggab':{'ablations':{'use_ggab':False}},'without_hmf':{'ablations':{'use_hmf':False}},
 'without_fgss':{'fgss':False},'without_earl':{'earl':False},'without_lesion':{'lesion':False}}
results=[]
for name,options in variants.items():
 for seed in a.seeds:
    opts=options.copy()
    if 'ablations' in opts:opts['ablations']={**{'use_mlge':True,'use_ggab':True,'use_hmf':True},**opts['ablations']}
    results.append({'variant':name,**train_experiment(a.manifest,str(Path(a.output)/name/str(seed)),seed=seed,epochs=a.epochs,batch=a.batch,pretrained=not a.no_pretrained,**opts)})
Path(a.output).mkdir(exist_ok=True,parents=True);(Path(a.output)/'summary.json').write_text(json.dumps(results,indent=2))
