"""Severity-conditioned corruption suite shared by all comparative models."""
import numpy as np
import torch
import torch.nn.functional as F
from ..data.preprocessing import denormalize,MEAN,STD

def corrupt(x,kind,severity=1,seed=42):
    """Deterministic relative to x, kind, severity, seed. severities 1..5."""
    if severity not in range(1,6):raise ValueError('severity must be 1..5')
    rgb=denormalize(x).clone();b,c,h,w=rgb.shape
    strength=severity/5
    gen=torch.Generator(device=x.device);gen.manual_seed(seed)
    if kind=='illumination':rgb=(rgb*(1-.65*strength)).clamp(0,1)
    elif kind=='shadow':
        col=torch.linspace(0,1,w,device=x.device).reshape(1,1,1,w)
        rgb=rgb*(1-.75*strength*col)
    elif kind=='blur':
        k=1+severity*2
        rgb=F.avg_pool2d(F.pad(rgb,(k//2,)*4,mode='reflect'),k,1)
    elif kind=='occlusion':
        side=max(1,int(min(h,w)*(.08+.3*strength)))
        for i in range(b):
            y=int(torch.randint(0,h-side+1,(1,),generator=gen,device=x.device))
            z=int(torch.randint(0,w-side+1,(1,),generator=gen,device=x.device))
            rgb[i,:,y:y+side,z:z+side]=.5
    elif kind=='scale':
        factor=1+.25*strength
        theta=torch.tensor([[[factor,0,0],[0,factor,0]]],device=x.device,dtype=x.dtype).repeat(b,1,1)
        rgb=F.grid_sample(rgb,F.affine_grid(theta,rgb.shape,align_corners=False),align_corners=False,padding_mode='border')
    elif kind=='background':
        # No target segmentation required: fixed border perturbation, not actual leaf segmentation.
        border=max(1,int(min(h,w)*(.05+.15*strength)))
        noise=torch.rand(rgb.shape,generator=gen,device=x.device,dtype=rgb.dtype)
        rgb[:,:,:border,:]=noise[:,:,:border,:];rgb[:,:,-border:,:]=noise[:,:,-border:,:]
        rgb[:,:,:,:border]=noise[:,:,:,:border];rgb[:,:,:,-border:]=noise[:,:,:,-border:]
    else:raise ValueError(f'Unknown corruption {kind}')
    mean=x.new_tensor(MEAN).view(1,3,1,1);std=x.new_tensor(STD).view(1,3,1,1)
    return (rgb.clamp(0,1)-mean)/std
