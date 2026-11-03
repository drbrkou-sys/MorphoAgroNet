"""Dataset-specific reproducible train / evaluation engine."""
import os,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from ..data.dataset import DiseaseDataset,class_names
from ..data.preprocessing import ImageTransform
from ..data.pseudo_masks import batch_pseudo_masks
from ..models.morphoagronet import MorphoAgroNet
from ..models.baselines import build_baseline
from ..evaluation.metrics import classification_metrics
from .losses import PrototypeMemory,fgss_loss,lesion_loss,earl_loss
from .earl import perturb

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def device_for(name='auto'):
    return torch.device('cuda' if name=='auto' and torch.cuda.is_available() else ('cpu' if name=='auto' else name))

def make_loaders(manifest,batch=32,workers=0,field_aware=True,size=224):
    datasets={k:DiseaseDataset(manifest,k,ImageTransform(k=='train',size,field_aware)) for k in ('train','val','test')}
    return {k:DataLoader(v,batch_size=batch,shuffle=k=='train',num_workers=workers,
                         pin_memory=torch.cuda.is_available()) for k,v in datasets.items()}

def get_model(name,num_classes,pretrained=True,ablations=None):
    if name=='morphoagronet':return MorphoAgroNet(num_classes,pretrained,**(ablations or {}))
    return build_baseline(name,num_classes,pretrained)

def prediction_pass(model,loader,device):
    model.eval();ys=[];ps=[];probs=[]
    with torch.no_grad():
        for x,y in loader:
            o=model(x.to(device));o=o['logits'] if isinstance(o,dict) else o
            p=o.softmax(1).cpu().numpy()
            ys.extend(y.numpy().tolist());ps.extend(p.argmax(1).tolist());probs.extend(p.tolist())
    result=classification_metrics(ys,ps,probs)
    return result,{'true':ys,'pred':ps,'prob':probs}

def atomic_save(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=str(path)+'.tmp';torch.save(obj,tmp);os.replace(tmp,path)

def load_checkpoint(path,device='cpu',pretrained=False):
    state=torch.load(path,map_location=device,weights_only=False)
    model=get_model(state['model_name'],len(state['class_names']),pretrained, state.get('ablations'))
    model.load_state_dict(state['model']);model.to(device).eval()
    return model,state

def train_experiment(manifest,output='results/run',name='morphoagronet',seed=42,epochs=100,
                     batch=32,workers=0,lr=1e-4,weight_decay=1e-4,patience=15,pretrained=True,
                     device='auto',earl=True,fgss=True,lesion=True,warmup=5,
                     ablations=None,amp=True,gradient_accumulation=1):
    seed_everything(seed);dev=device_for(device)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    names=class_names(manifest);loaders=make_loaders(manifest,batch,workers)
    model=get_model(name,len(names),pretrained,ablations).to(dev)
    is_morpho=isinstance(model,MorphoAgroNet) and name=='morphoagronet'
    if is_morpho:model.freeze_lower(True)
    prototypes=PrototypeMemory(len(names)).to(dev) if is_morpho else None
    low=[];high=[]
    for key,p in model.named_parameters():
        if is_morpho and (key.startswith('stem.') or key.startswith('layer')):low.append(p)
        else:high.append(p)
    groups=[{'params':low,'lr':lr*.1,'scale':.1},{'params':high,'lr':lr,'scale':1.}] if low else [{'params':high,'lr':lr,'scale':1.}]
    optim=torch.optim.AdamW(groups,weight_decay=weight_decay)
    scaler=torch.amp.GradScaler('cuda',enabled=amp and dev.type=='cuda')
    healthy_ids=[i for i,s in enumerate(names) if 'healthy' in s.lower() or s.lower() in ('apple leaf','tomato leaf','bell_pepper leaf')]
    best=-1;bad=0;history=[]
    for ep in range(epochs):
        model.train()
        if is_morpho and ep==warmup:model.freeze_lower(False)
        if is_morpho and ep<warmup:model.freeze_lower(True)
        # 5-epoch warmup followed by cosine decay
        if ep<warmup:factor=(ep+1)/max(1,warmup)
        else:factor=.5*(1+np.cos(np.pi*(ep-warmup)/max(1,epochs-warmup)))
        for pg in optim.param_groups:pg['lr']=max(1e-6,lr*pg['scale']*factor)
        loss_total=0.;nsteps=0;optim.zero_grad(set_to_none=True)
        for step,(x,y) in enumerate(loaders['train']):
            x=x.to(dev);y=y.to(dev)
            with torch.autocast(device_type=dev.type,enabled=amp and dev.type=='cuda'):
                out=model(x,return_details=True) if is_morpho else None
                logits=out['logits'] if is_morpho else model(x)
                loss=nn.functional.cross_entropy(logits,y,label_smoothing=.1)
                if is_morpho:
                    if lesion:
                        weak=batch_pseudo_masks(x,y,healthy_ids)
                        loss=loss+.2*lesion_loss(out['soft_lesion'],weak)
                    if fgss:
                        prototypes.update(out['normalized_embedding'],y)
                        fg,_=fgss_loss(out['normalized_embedding'],y,prototypes)
                        loss=loss+.1*fg
                    if earl:
                        px=perturb(x,lesion=out['binary_lesion'])
                        pout=model(px,return_details=True)
                        cons,_=earl_loss(out,pout)
                        loss=loss+.2*cons
                loss=loss/gradient_accumulation
            scaler.scale(loss).backward()
            if (step+1)%gradient_accumulation==0 or (step+1)==len(loaders['train']):
                scaler.unscale_(optim);nn.utils.clip_grad_norm_(model.parameters(),5.0)
                scaler.step(optim);scaler.update();optim.zero_grad(set_to_none=True)
            loss_total+=float(loss.detach())*gradient_accumulation;nsteps+=1
        val,_=prediction_pass(model,loaders['val'],dev)
        score=val['macro_f1']
        entry={'epoch':ep+1,'train_loss':loss_total/max(1,nsteps),'val_macro_f1':score,'val_accuracy':val['accuracy']}
        history.append(entry);print(json.dumps(entry),flush=True)
        if score>best+1e-8:
            best=score;bad=0
            atomic_save({'model':model.state_dict(),'model_name':name,'class_names':names,
                         'seed':seed,'epoch':ep+1,'val_f1':best,'ablations':ablations,
                         'config':{'lr':lr,'batch':batch,'warmup':warmup,'epochs':epochs,'earl':earl,'fgss':fgss,'lesion':lesion}},output/'best.pt')
        else:bad+=1
        if bad>=patience:break
    loaded=torch.load(output/'best.pt',map_location=dev,weights_only=False)
    model.load_state_dict(loaded['model']);test,prediction=prediction_pass(model,loaders['test'],dev)
    summary={'model':name,'seed':seed,'best_epoch':loaded['epoch'],'validation_macro_f1':best,'test':test,
             'classes':names,'training_epochs':len(history)}
    (output/'history.json').write_text(json.dumps(history,indent=2))
    (output/'metrics.json').write_text(json.dumps(summary,indent=2))
    (output/'predictions.json').write_text(json.dumps(prediction))
    return summary
