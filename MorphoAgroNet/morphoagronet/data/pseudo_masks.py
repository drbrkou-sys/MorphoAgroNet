"""Unsupervised CIELAB color-deviation weak lesion pseudo-labels."""
import cv2
import numpy as np
import torch
from .preprocessing import denormalize

def pseudo_mask_rgb(img,healthy=False):
    h,w=img.shape[:2]
    if healthy:return np.zeros((h,w),dtype=np.float32)
    rgb=np.uint8(np.clip(img,0,255));lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    c=rgb.astype(np.float32)/255
    exg=2*c[:,:,1]-c[:,:,0]-c[:,:,2]
    exg=((exg-exg.min())/(np.ptp(exg)+1e-6)*255).astype(np.uint8)
    _,leaf=cv2.threshold(exg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if (leaf>0).mean()<.05:leaf[:]=255
    chrom=lab[:,:,1:].astype(np.float32)
    med=np.median(chrom[leaf>0],axis=0)
    dist=np.linalg.norm(chrom-med,axis=2)
    valid=np.uint8(np.clip(dist/(dist.max()+1e-6)*255,0,255))
    level,_=cv2.threshold(valid[leaf>0].reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask=np.uint8((valid>level)&(leaf>0))*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    return (mask/255).astype(np.float32)

def batch_pseudo_masks(images,labels,healthy_ids=None):
    healthy=set(healthy_ids or [])
    imgs=(denormalize(images.detach()).cpu().permute(0,2,3,1).numpy()*255).astype(np.uint8)
    masks=np.stack([pseudo_mask_rgb(im,int(y) in healthy) for im,y in zip(imgs,labels)])
    return torch.from_numpy(masks).to(images.device).unsqueeze(1)
