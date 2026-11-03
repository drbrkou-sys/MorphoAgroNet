"""Independent lesion mask validation; CSV must contain image,mask,class_id."""
import argparse,json,csv
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from morphoagronet.training.trainer import load_checkpoint,device_for
from morphoagronet.data.preprocessing import ImageTransform
from morphoagronet.evaluation.explainability import explanation,localization_scores,boundary_f1
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--pairs',required=True,help='CSV columns: image,mask,class_id')
p.add_argument('--output',default='results/localization.json');p.add_argument('--device',default='auto')
a=p.parse_args();dev=device_for(a.device);model,_=load_checkpoint(a.checkpoint,dev)
if not hasattr(model,'lesion'):raise ValueError('MorphoAgroNet checkpoint required')
rows=[]
with open(a.pairs,newline='') as f:
 for line in csv.DictReader(f):
    im=Image.open(line['image']).convert('RGB');mask=Image.open(line['mask']).convert('L').resize((224,224),Image.Resampling.NEAREST)
    x=ImageTransform()(im).unsqueeze(0).to(dev)
    target=torch.tensor([int(line['class_id'])],device=dev)
    maps=explanation(model,x,classes=target)
    heat=maps['integrated'][0,0].cpu().numpy();gt=np.asarray(mask)>127
    r=localization_scores(heat,gt)
    r['boundary_f1']=boundary_f1(heat>np.quantile(heat,.75),gt)
    rows.append({'image':line['image'],**r})
keys=('iou','dice','pointing_game','energy_pointing','boundary_f1')
res={'n':len(rows),'mean':{k:float(np.mean([z[k] for z in rows])) if rows else None for k in keys},'per_image':rows}
Path(a.output).parent.mkdir(exist_ok=True,parents=True);Path(a.output).write_text(json.dumps(res,indent=2));print(json.dumps(res['mean'],indent=2))
