"""Exact stop-gradient contour descriptors and differentiable multi-scale soft-map encoding."""
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# 4 curvature statistics, irregularity, extent, asymmetry, spot density = 8
DESCRIPTOR_DIM=8

def descriptor_from_mask(raw):
    mask=np.uint8(raw>0)
    h,w=mask.shape;area=int(mask.sum())
    if area<3:return np.zeros(DESCRIPTOR_DIM,dtype=np.float32)
    n,comps,stats,_=cv2.connectedComponentsWithStats(mask,8)
    valid=[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>=max(1,int(.001*h*w))]
    count=len(valid)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    perim=sum(cv2.arcLength(c,True) for c in contours)
    irregularity=perim*perim/(4*np.pi*area+1e-6)
    curv=[]
    for c in contours:
        pts=c[:,0,:].astype(np.float32)
        if len(pts)<5:continue
        d1=(np.roll(pts,-1,axis=0)-np.roll(pts,1,axis=0))/2
        d2=np.roll(pts,-1,axis=0)-2*pts+np.roll(pts,1,axis=0)
        v=np.abs(d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0])/(np.sum(d1*d1,axis=1)**1.5+1e-4)
        curv.extend(np.clip(v,0,10).tolist())
    cm=np.array(curv) if curv else np.zeros(1)
    ys,xs=np.nonzero(mask)
    mid=np.median(xs);left=(xs<mid).sum();right=(xs>=mid).sum()
    asym=abs(left-right)/max(len(xs),1)
    # density normalized by image area, not mistaken for number of lesions
    vals=[np.mean(cm),np.std(cm),np.quantile(cm,.25),np.quantile(cm,.75),
          np.log1p(irregularity),area/(h*w),asym,count/(h*w)]
    return np.nan_to_num(np.array(vals,dtype=np.float32),nan=0,posinf=0,neginf=0)

def batch_exact_geometry(soft,threshold=True):
    xs=soft.detach().float().cpu().numpy()[:,0]
    out=[];masks=[]
    for x in xs:
        u=np.uint8(np.clip(x*255,0,255))
        if threshold:
            _,m=cv2.threshold(u,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else:m=np.uint8(x>.5)*255
        m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        n,cc,stats,_=cv2.connectedComponentsWithStats(m,8)
        for k in range(1,n):
            if stats[k,cv2.CC_STAT_AREA]<max(1,int(.001*m.size)):m[cc==k]=0
        out.append(descriptor_from_mask(m));masks.append(m.astype(np.float32)/255)
    return (torch.tensor(np.stack(out),device=soft.device,dtype=soft.dtype),
            torch.tensor(np.stack(masks),device=soft.device,dtype=soft.dtype).unsqueeze(1))

class MLGE(nn.Module):
    def __init__(self,dim=128):
        super().__init__()
        self.encoders=nn.ModuleList([nn.Sequential(nn.Conv2d(1,32,k,padding=k//2),nn.ReLU(),
                           nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1)) for k in (3,5,7)])
        self.project=nn.ModuleList([nn.Sequential(nn.Linear(64+DESCRIPTOR_DIM,dim),nn.ReLU()) for _ in range(3)])
        self.score=nn.Linear(dim,1)
    def forward(self,soft,descriptor=None):
        if descriptor is None:descriptor,_=batch_exact_geometry(soft)
        reps=[]
        for i,(enc,proj) in enumerate(zip(self.encoders,self.project)):
            scale=F.avg_pool2d(soft,2**i,2**i) if i else soft
            feat=enc(scale).flatten(1)
            reps.append(proj(torch.cat([feat,descriptor],dim=1)))
        stack=torch.stack(reps,1)
        weights=self.score(stack).squeeze(-1).softmax(1)
        return (stack*weights.unsqueeze(-1)).sum(1),weights
