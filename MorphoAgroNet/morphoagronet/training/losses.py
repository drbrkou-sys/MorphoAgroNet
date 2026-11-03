"""Lesion, geometry, FGSS, and EARL training objectives."""
import torch
from torch import nn
import torch.nn.functional as F
from ..data.preprocessing import denormalize,sobel_edges

class PrototypeMemory(nn.Module):
    def __init__(self,num_classes,dim=256,momentum=.9):
        super().__init__();self.momentum=momentum
        self.register_buffer('centers',torch.zeros(num_classes,dim))
        self.register_buffer('seen',torch.zeros(num_classes,dtype=torch.bool))
    @torch.no_grad()
    def update(self,z,y):
        for cls in y.unique():
            c=int(cls)
            center=F.normalize(z[y==cls].detach().mean(0,keepdim=True),dim=1)[0]
            if not self.seen[c]:self.centers[c]=center
            else:self.centers[c]=F.normalize(self.momentum*self.centers[c]+(1-self.momentum)*center,dim=0)
            self.seen[c]=True

def fgss_loss(z,y,prototypes,margin=.4,hard_margin=.25):
    device=z.device;zero=z.sum()*0
    centers=prototypes.centers
    valid=prototypes.seen[y]
    intra=(z[valid]-centers[y[valid]]).pow(2).sum(1).mean() if valid.any() else zero
    p=centers[prototypes.seen]
    if len(p)>1:
        distances=torch.cdist(p,p)
        off=~torch.eye(len(p),device=device,dtype=torch.bool)
        inter=F.relu(margin-distances[off]).square().mean()
    else:inter=zero
    if len(z)>1:
        dist=torch.cdist(z,z)
        allowed=y[:,None]!=y[None,:]
        hardest=dist.masked_fill(~allowed,float('inf')).min(1).values
        positive=(z-centers[y]).norm(dim=1)
        has=allowed.any(1)&valid
        hard=F.relu(positive[has]-hardest[has]+hard_margin).mean() if has.any() else zero
    else:hard=zero
    return intra+inter+hard,{'fgss_intra':intra,'fgss_inter':inter,'fgss_hard':hard}

def lesion_loss(pred,target,area_weight=.1):
    target=F.interpolate(target,size=pred.shape[-2:],mode='nearest')
    bce=F.binary_cross_entropy(pred.clamp(1e-5,1-1e-5),target)
    area=(pred.mean((1,2,3))-target.mean((1,2,3))).abs().mean()
    return bce+area_weight*area

def geometry_preservation_loss(clean,corrected):
    return F.l1_loss(sobel_edges(denormalize(clean)),sobel_edges(denormalize(corrected)))

def earl_loss(clean,perturbed,rep_weight=1.,pred_weight=1.):
    a=clean['normalized_embedding'];b=perturbed['normalized_embedding']
    rep=F.mse_loss(a,b)
    logp=F.log_softmax(clean['logits'],1);logq=F.log_softmax(perturbed['logits'],1)
    p=logp.exp();q=logq.exp()
    kl=.5*(F.kl_div(logp,q,reduction='batchmean')+F.kl_div(logq,p,reduction='batchmean'))
    return rep_weight*rep+pred_weight*kl,{'earl_rep':rep,'earl_kl':kl}
