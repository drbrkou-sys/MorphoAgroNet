from pathlib import Path as _ProjectPath
import sys as _project_sys
_project_sys.path.insert(0,str(_ProjectPath(__file__).resolve().parents[1]))
import argparse,json
from morphoagronet.training.trainer import train_experiment
p=argparse.ArgumentParser()
p.add_argument('--manifest',required=True);p.add_argument('--output',default='results/morpho')
p.add_argument('--model',default='morphoagronet');p.add_argument('--seed',type=int,default=42)
p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch',type=int,default=32)
p.add_argument('--workers',type=int,default=0);p.add_argument('--lr',type=float,default=1e-4)
p.add_argument('--patience',type=int,default=15);p.add_argument('--device',default='auto')
p.add_argument('--no-pretrained',action='store_true');p.add_argument('--no-earl',action='store_true')
p.add_argument('--no-fgss',action='store_true');p.add_argument('--no-lesion',action='store_true')
p.add_argument('--no-mlge',action='store_true');p.add_argument('--no-ggab',action='store_true')
p.add_argument('--no-hmf',action='store_true');p.add_argument('--grad-accum',type=int,default=1)
a=p.parse_args()
ablations={k:not getattr(a,'no_'+k) for k in ('mlge','ggab','hmf')}
r=train_experiment(a.manifest,a.output,a.model,a.seed,a.epochs,a.batch,a.workers,a.lr,
                   patience=a.patience,pretrained=not a.no_pretrained,device=a.device,
                   earl=not a.no_earl,fgss=not a.no_fgss,lesion=not a.no_lesion,
                   ablations=ablations if a.model=='morphoagronet' else None,
                   gradient_accumulation=a.grad_accum)
print(json.dumps(r,indent=2))
