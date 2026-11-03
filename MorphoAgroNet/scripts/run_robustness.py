from pathlib import Path as _ProjectPath
import sys as _project_sys
_project_sys.path.insert(0,str(_ProjectPath(__file__).resolve().parents[1]))
import argparse,json
from pathlib import Path
import torch
from morphoagronet.evaluation.robustness import corrupt
from morphoagronet.evaluation.metrics import classification_metrics
from morphoagronet.training.trainer import load_checkpoint,make_loaders,device_for,prediction_pass
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
p.add_argument('--output',default='results/robustness.json');p.add_argument('--device',default='auto');p.add_argument('--batch',type=int,default=16)
a=p.parse_args();dev=device_for(a.device);model,_=load_checkpoint(a.checkpoint,dev)
loader=make_loaders(a.manifest,a.batch)['test'];clean,_=prediction_pass(model,loader,dev)
rows=[]
for kind in ('shadow','blur','occlusion','scale','background','illumination'):
 for severity in range(1,6):
    ys=[];pred=[];prob=[];model.eval()
    with torch.no_grad():
     for bi,(x,y) in enumerate(loader):
        out=model(corrupt(x.to(dev),kind,severity,42+bi));out=out['logits'] if isinstance(out,dict) else out
        pro=out.softmax(1).cpu().numpy();ys+=y.tolist();pred+=pro.argmax(1).tolist();prob+=pro.tolist()
    metric=classification_metrics(ys,pred,prob)
    rows.append({'corruption':kind,'severity':severity,**metric,
                 'macro_f1_degradation_pp':100*(clean['macro_f1']-metric['macro_f1'])})
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps({'clean':clean,'corruptions':rows},indent=2))
print('Saved',a.output)
