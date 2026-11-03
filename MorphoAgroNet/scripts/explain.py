from pathlib import Path as _ProjectPath
import sys as _project_sys
_project_sys.path.insert(0,str(_ProjectPath(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from morphoagronet.training.trainer import load_checkpoint,device_for
from morphoagronet.data.preprocessing import ImageTransform
from morphoagronet.evaluation.explainability import explanation
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True);p.add_argument('--output',default='figures/explanation');p.add_argument('--device',default='auto')
a=p.parse_args();dev=device_for(a.device);model,state=load_checkpoint(a.checkpoint,dev)
x=ImageTransform()(Image.open(a.image).convert('RGB')).unsqueeze(0).to(dev)
if not hasattr(model,'lesion'):raise ValueError('Explainability module requires MorphoAgroNet checkpoint')
r=explanation(model,x);Path(a.output).mkdir(parents=True,exist_ok=True)
for k in ('gradcam','attention','integrated','lesion'):
    mat=r[k][0,0].cpu().numpy();im=Image.fromarray(np.uint8(np.clip(mat,0,1)*255))
    im.save(str(Path(a.output)/(k+'.png')))
print('Prediction:',state['class_names'][int(r['classes'][0])]);print('Saved to:',a.output)
