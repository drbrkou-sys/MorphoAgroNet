from pathlib import Path as _ProjectPath
import sys as _project_sys
_project_sys.path.insert(0,str(_ProjectPath(__file__).resolve().parents[1]))
import argparse,json
from PIL import Image
import torch
from morphoagronet.data.preprocessing import ImageTransform
from morphoagronet.training.trainer import load_checkpoint,device_for
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True);p.add_argument('--device',default='auto')
a=p.parse_args();dev=device_for(a.device);model,state=load_checkpoint(a.checkpoint,dev)
x=ImageTransform()(Image.open(a.image).convert('RGB')).unsqueeze(0).to(dev)
with torch.no_grad(): probabilities=model(x).softmax(1)[0]
i=int(probabilities.argmax());print(json.dumps({'class':state['class_names'][i],'confidence':float(probabilities[i]),'probabilities':{k:float(v) for k,v in zip(state['class_names'],probabilities)}},indent=2))
