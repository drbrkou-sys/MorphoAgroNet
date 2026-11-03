from pathlib import Path as _ProjectPath
import sys as _project_sys
_project_sys.path.insert(0,str(_ProjectPath(__file__).resolve().parents[1]))
import argparse,json
from pathlib import Path
from morphoagronet.training.trainer import load_checkpoint,make_loaders,prediction_pass,device_for
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
p.add_argument('--batch',type=int,default=32);p.add_argument('--device',default='auto');p.add_argument('--output',default='results/evaluation.json')
a=p.parse_args();dev=device_for(a.device);m,_=load_checkpoint(a.checkpoint,dev)
metrics,pred=prediction_pass(m,make_loaders(a.manifest,a.batch)['test'],dev)
Path(a.output).parent.mkdir(exist_ok=True,parents=True);Path(a.output).write_text(json.dumps({'metrics':metrics,'predictions':pred},indent=2))
print(json.dumps(metrics,indent=2))
