"""Grad-CAM on fused map and geometry-conditioned attention fusion."""
import numpy as np
import torch
import torch.nn.functional as F
import cv2

def normalize_map(x):
    x=x-x.amin(dim=(-2,-1),keepdim=True)
    return x/(x.amax(dim=(-2,-1),keepdim=True)+1e-8)

def explanation(model,x,classes=None):
    """On-demand explanation. Inference requires gradient-enabled context."""
    model.eval()
    with torch.enable_grad():
        out=model(x,return_details=True)
        logits=out['logits'];labels=logits.argmax(1) if classes is None else classes
        fmap=out['fused_map'];grad=torch.autograd.grad(logits.gather(1,labels.view(-1,1)).sum(),fmap,retain_graph=False)[0]
        weights=grad.mean(dim=(2,3),keepdim=True)
        cam=F.relu((fmap*weights).sum(1,keepdim=True))
        cam=normalize_map(F.interpolate(cam,size=x.shape[-2:],mode='bilinear',align_corners=False))
        maps=[F.interpolate(a,size=x.shape[-2:],mode='bilinear',align_corners=False) for a in out['attention_maps']]
        stacked=torch.stack(maps,dim=1)
        wa=out['hierarchy_weights'][:,:,None,None,None]
        attention=normalize_map((stacked*wa).sum(1))
        agreement=1-(cam-attention).abs().mean(dim=(1,2,3),keepdim=True)
        alpha=torch.sigmoid(4*(agreement-.5))
        combined=normalize_map(alpha*cam+(1-alpha)*attention)
        return {'classes':labels.detach(),'probabilities':logits.softmax(1).detach(),
                'gradcam':cam.detach(),'attention':attention.detach(),
                'integrated':combined.detach(),'lesion':out['binary_lesion'].detach()}

def localization_scores(heat,gt):
    heat=np.asarray(heat).squeeze().astype(np.float32)
    gt=np.asarray(gt).squeeze()>0
    if heat.shape!=gt.shape:heat=cv2.resize(heat,(gt.shape[1],gt.shape[0]))
    threshold=np.quantile(heat,.75);pred=heat>threshold
    inter=int(np.logical_and(pred,gt).sum());union=int(np.logical_or(pred,gt).sum())
    point=bool(gt[np.unravel_index(heat.argmax(),heat.shape)])
    energy=float(heat[gt].sum()/(heat.sum()+1e-8))
    return {'iou':inter/(union+1e-8),'dice':2*inter/(pred.sum()+gt.sum()+1e-8),
            'pointing_game':float(point),'energy_pointing':energy}

def boundary_f1(pred,gt,tolerance=2):
    pred=np.uint8(np.asarray(pred)>0);gt=np.uint8(np.asarray(gt)>0)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    a=pred-cv2.erode(pred,kernel);b=gt-cv2.erode(gt,kernel)
    dil=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*tolerance+1,)*2)
    p=(a*cv2.dilate(b,dil)).sum()/(a.sum()+1e-8)
    r=(b*cv2.dilate(a,dil)).sum()/(b.sum()+1e-8)
    return float(2*p*r/(p+r+1e-8))
