from pathlib import Path as _ProjectPath
import sys as _project_sys
_project_sys.path.insert(0,str(_ProjectPath(__file__).resolve().parents[1]))
import argparse,json,time
import torch
from morphoagronet.training.trainer import load_checkpoint,device_for
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='auto');p.add_argument('--repeats',type=int,default=20)
a=p.parse_args();dev=device_for(a.device);model,_=load_checkpoint(a.checkpoint,dev)
x=torch.rand(1,3,224,224,device=dev)
def sync():
 if dev.type=='cuda':torch.cuda.synchronize()
with torch.inference_mode():
 for _ in range(3):model(x)
 sync();before=time.perf_counter()
 for _ in range(a.repeats):model(x)
 sync();ms=1000*(time.perf_counter()-before)/a.repeats
n=sum(t.numel() for t in model.parameters());print(json.dumps({'parameters':n,'fp32_parameter_mb':n*4/1e6,'latency_ms_batch1':ms,'throughput_images_per_second':1000/ms,'device':str(dev)},indent=2))
