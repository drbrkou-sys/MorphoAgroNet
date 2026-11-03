from pathlib import Path as _ProjectPath
import sys as _project_sys
_project_sys.path.insert(0,str(_ProjectPath(__file__).resolve().parents[1]))
"""Source-only transfer and optional 10% target training adaptation on compatible classes."""
import argparse,json,copy,random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from morphoagronet.training.trainer import load_checkpoint,device_for
from morphoagronet.data.dataset import DiseaseDataset,class_names
from morphoagronet.data.preprocessing import ImageTransform
from morphoagronet.evaluation.transfer import mappings
from morphoagronet.evaluation.metrics import classification_metrics
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--target-manifest',required=True)
p.add_argument('--source-kind',required=True,choices=['plantvillage','plantdoc','pathology2020'])
p.add_argument('--target-kind',required=True,choices=['plantvillage','plantdoc','pathology2020'])
p.add_argument('--adapt',action='store_true');p.add_argument('--epochs',type=int,default=30);p.add_argument('--seed',type=int,default=42)
p.add_argument('--device',default='auto');p.add_argument('--output',default='results/transfer.json')
a=p.parse_args();dev=device_for(a.device);model,state=load_checkpoint(a.checkpoint,dev)
target_names=class_names(a.target_manifest)
pairs=mappings(state['class_names'],target_names,a.source_kind,a.target_kind)
src=[q[0] for q in pairs];tgt=[q[1] for q in pairs]
translate={v:i for i,v in enumerate(tgt)}
train=DiseaseDataset(a.target_manifest,'train',ImageTransform(True));test=DiseaseDataset(a.target_manifest,'test',ImageTransform(False))

def filter_loader(ds,batch=16,subset_fraction=1):
    indices=[i for i,r in ds.df.iterrows() if int(r.class_id) in translate]
    if subset_fraction<1:
        random.seed(a.seed);picked=[]
        for cls in tgt:
            matching=[i for i in indices if int(ds.df.iloc[i].class_id)==cls]
            count=min(len(matching),max(2,int(round(len(matching)*subset_fraction))))
            picked+=random.sample(matching,count)
        indices=picked
    return DataLoader(Subset(ds,indices),batch_size=batch,shuffle=ds is train)

trainloader=filter_loader(train,.0 if False else 16,.1);testloader=filter_loader(test)

def get_logits(x):
    out=model(x);return (out['logits'] if isinstance(out,dict) else out)[:,src]

def eval_model():
    model.eval();y=[];p=[];prob=[]
    with torch.no_grad():
        for x,t in testloader:
            z=get_logits(x.to(dev)).softmax(1).cpu().numpy()
            y.extend([translate[int(q)] for q in t]);p.extend(z.argmax(1));prob.extend(z.tolist())
    return classification_metrics(y,p,prob),len(y)
source,n_test=eval_model();result={'shared_classes':[q[2] for q in pairs],'source_only':source,'target_test_images':n_test}
if a.adapt:
    model.train();opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    for epoch in range(a.epochs):
        for x,y in trainloader:
            x=x.to(dev);y=torch.tensor([translate[int(q)] for q in y],device=dev)
            loss=nn.functional.cross_entropy(get_logits(x),y)
            opt.zero_grad();loss.backward();opt.step()
    model.eval();result['adapted'],_=eval_model();result['adaptation_images']=len(trainloader.dataset)
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
