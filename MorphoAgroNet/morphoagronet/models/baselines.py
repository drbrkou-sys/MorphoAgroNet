"""ImageNet pretrained baseline architectures and controlled backbone-only architecture."""
import torch
from torch import nn
from torchvision import models
from .morphoagronet import MorphoAgroNet

def build_baseline(name,num_classes,pretrained=True):
    name=name.lower().replace('-','_')
    if name=='backbone_only':return MorphoAgroNet(num_classes,pretrained,use_mlge=False,use_ggab=False,use_hmf=False)
    specs={
      'resnet50':(models.resnet50,models.ResNet50_Weights.IMAGENET1K_V2,'fc'),
      'densenet121':(models.densenet121,models.DenseNet121_Weights.IMAGENET1K_V1,'classifier'),
      'efficientnet_b0':(models.efficientnet_b0,models.EfficientNet_B0_Weights.IMAGENET1K_V1,'classifier'),
      'convnext_tiny':(models.convnext_tiny,models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,'classifier'),
      'mobilenet_v3_large':(models.mobilenet_v3_large,models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,'classifier'),
      'vit_b_16':(models.vit_b_16,models.ViT_B_16_Weights.IMAGENET1K_V1,'heads'),
      'swin_t':(models.swin_t,models.Swin_T_Weights.IMAGENET1K_V1,'head'),
    }
    if name not in specs:raise ValueError(f'Unknown model {name}; choose {list(specs)} or backbone_only')
    factory,weights,head=specs[name]; net=factory(weights=weights if pretrained else None)
    if head=='fc':net.fc=nn.Linear(net.fc.in_features,num_classes)
    elif head=='head':net.head=nn.Linear(net.head.in_features,num_classes)
    elif head=='heads':net.heads.head=nn.Linear(net.heads.head.in_features,num_classes)
    else:
        old=getattr(net,head)
        if isinstance(old,nn.Sequential):
            idx=max(i for i,l in enumerate(old) if isinstance(l,nn.Linear))
            old[idx]=nn.Linear(old[idx].in_features,num_classes)
        else:setattr(net,head,nn.Linear(old.in_features,num_classes))
    return net
