from pathlib import Path as _ProjectPath
import sys as _project_sys
_project_sys.path.insert(0,str(_ProjectPath(__file__).resolve().parents[1]))
import argparse,json
from pathlib import Path
from morphoagronet.training.trainer import train_experiment
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',default='results/baselines')
p.add_argument('--models',nargs='+',default=['resnet50','densenet121','efficientnet_b0','convnext_tiny','mobilenet_v3_large','vit_b_16','swin_t','backbone_only'])
p.add_argument('--seeds',nargs='+',type=int,default=[42,123,2024,2025,3407]);p.add_argument('--epochs',type=int,default=100)
p.add_argument('--batch',type=int,default=32);p.add_argument('--no-pretrained',action='store_true')
a=p.parse_args();results=[]
for model in a.models:
    for seed in a.seeds:
        target=str(Path(a.output)/model/str(seed));results.append(train_experiment(a.manifest,target,name=model,seed=seed,epochs=a.epochs,batch=a.batch,pretrained=not a.no_pretrained))
Path(a.output).mkdir(exist_ok=True,parents=True);(Path(a.output)/'summary.json').write_text(json.dumps(results,indent=2))
