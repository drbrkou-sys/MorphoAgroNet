"""Geometry-friendly image enhancement and standardized tensor transforms."""
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF

MEAN=[.485,.456,.406];STD=[.229,.224,.225]

class FieldAwarePreprocess:
    def __init__(self,size=224,clahe_clip=1.5): self.size=size;self.clip=clahe_clip
    def __call__(self,image):
        a=np.asarray(image.convert('RGB').resize((self.size,self.size),Image.Resampling.BILINEAR))
        lab=cv2.cvtColor(a,cv2.COLOR_RGB2LAB); l,aa,b=cv2.split(lab)
        l=cv2.createCLAHE(clipLimit=self.clip,tileGridSize=(8,8)).apply(l)
        enhanced=cv2.cvtColor(cv2.merge([l,aa,b]),cv2.COLOR_LAB2RGB)
        enhanced=cv2.bilateralFilter(enhanced,3,20,20)
        return Image.fromarray(enhanced)

class ImageTransform:
    def __init__(self,train=False,size=224,field_aware=True):
        self.pre=FieldAwarePreprocess(size) if field_aware else lambda im:im.resize((size,size))
        self.aug=T.Compose([T.RandomResizedCrop(size,scale=(.8,1.0)),T.RandomHorizontalFlip(),T.ColorJitter(.2,.2,.2,.05)]) if train else None
    def __call__(self,img):
        img=self.pre(img)
        if self.aug:img=self.aug(img)
        return TF.normalize(TF.to_tensor(img),MEAN,STD)

def denormalize(x):
    mean=x.new_tensor(MEAN).view(1,3,1,1);std=x.new_tensor(STD).view(1,3,1,1)
    return (x*std+mean).clamp(0,1)

def sobel_edges(x):
    import torch.nn.functional as F
    if x.shape[1]==3:x=x.mean(1,keepdim=True)
    kx=x.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).view(1,1,3,3)/8
    ky=kx.transpose(-1,-2)
    return (F.conv2d(x,kx,padding=1).square()+F.conv2d(x,ky,padding=1).square()+1e-8).sqrt()
