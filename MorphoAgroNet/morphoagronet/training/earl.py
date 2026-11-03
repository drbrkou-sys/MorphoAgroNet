"""Field-condition perturbation generation on denormalized tensors."""
import random
import torch
import torch.nn.functional as F
from ..data.preprocessing import denormalize,MEAN,STD

def perturb(x,severity=.5,lesion=None):
    """Sample field perturbations; operate in RGB [0,1], then ImageNet-normalize."""
    image=denormalize(x);b,c,h,w=image.shape
    for i in range(b):
        z=image[i:i+1]
        modes=random.sample(['shadow','blur','occlusion','scale','illumination','background'],k=random.randint(1,3))
        for mode in modes:
            if mode=='illumination':
                factor=1+random.uniform(-.45,.45)*severity;z=(z*factor).clamp(0,1)
            elif mode=='shadow':
                xx=torch.linspace(0,1,w,device=x.device).view(1,1,1,w)
                intensity=random.uniform(.1,.6)*severity
                z=z*(1-intensity*(xx if random.random()<.5 else 1-xx))
            elif mode=='blur':
                k=3 if severity<.6 else 5
                z=F.avg_pool2d(F.pad(z,(k//2,)*4,mode='reflect'),k,stride=1)
            elif mode=='occlusion':
                side=max(1,int(min(h,w)*random.uniform(.03,.12)*severity))
                y=random.randint(0,h-side);xx=random.randint(0,w-side)
                if lesion is not None:
                    mask=F.interpolate(lesion[i:i+1].detach(),size=(h,w),mode='bilinear',align_corners=False)
                    if mask[:,:,y:y+side,xx:xx+side].mean()>.25:continue
                z=z.clone();z[:,:,y:y+side,xx:xx+side]=z.mean(dim=(2,3),keepdim=True)
            elif mode=='scale':
                factor=1+random.uniform(-.15,.15)*severity
                theta=torch.tensor([[[factor,0,0],[0,factor,0]]],device=x.device,dtype=x.dtype)
                z=F.grid_sample(z,F.affine_grid(theta,z.shape,align_corners=False),padding_mode='border',align_corners=False)
            elif mode=='background':
                if lesion is not None:
                    mask=F.interpolate(lesion[i:i+1].detach(),size=(h,w),mode='bilinear',align_corners=False).clamp(0,1)
                    noise=torch.rand_like(z)*.3+.35
                    z=z*mask+(z*.65+noise*.35)*(1-mask)
        image[i:i+1]=z.clamp(0,1)
    mean=x.new_tensor(MEAN).view(1,3,1,1);std=x.new_tensor(STD).view(1,3,1,1)
    return (image-mean)/std
