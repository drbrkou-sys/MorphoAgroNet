"""MorphoAgroNet: weak lesion estimation -> MLGE -> GGAB -> HMF -> FGSS embedding."""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from .geometry import MLGE, batch_exact_geometry

class LesionEstimator(nn.Module):
    def __init__(self,input_channels=256):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(input_channels,128,3,padding=1),nn.ReLU(inplace=True),
                 nn.Conv2d(128,128,3,padding=1),nn.ReLU(inplace=True),
                 nn.Conv2d(128,1,1))
    def forward(self,x):return self.net(x).sigmoid()

class GGAB(nn.Module):
    def __init__(self,channels=256,geo_dim=128):
        super().__init__()
        self.geometry=nn.Linear(geo_dim,channels)
        self.interaction=nn.Sequential(nn.Conv2d(channels*2,channels,1),nn.ReLU(inplace=True),
                                       nn.Conv2d(channels,1,3,padding=1))
        self.gamma=nn.Parameter(torch.tensor(.1))
    def forward(self,feature,geometry):
        g=self.geometry(geometry)[:,:,None,None].expand_as(feature)
        attention=self.interaction(torch.cat([feature,g],dim=1)).sigmoid()
        return feature+self.gamma*feature*attention,attention

class HMF(nn.Module):
    def __init__(self,channels=256,embedding=256):
        super().__init__()
        self.scorer=nn.Sequential(nn.Linear(channels,64),nn.ReLU(),nn.Linear(64,1))
        self.refine=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1,bias=False),
                                  nn.BatchNorm2d(channels),nn.ReLU())
        self.proj=nn.Sequential(nn.Linear(channels,embedding),nn.ReLU())
    def forward(self,maps):
        scores=torch.stack([self.scorer(F.adaptive_avg_pool2d(x,1).flatten(1)) for x in maps],1).squeeze(-1)
        weights=scores.softmax(1)
        shape=maps[0].shape[-2:]
        resized=[F.interpolate(x,size=shape,mode='bilinear',align_corners=False) if x.shape[-2:]!=shape else x for x in maps]
        fused=sum(x*weights[:,i,None,None,None] for i,x in enumerate(resized))
        fused=self.refine(fused)
        embedding=self.proj(F.adaptive_avg_pool2d(fused,1).flatten(1))
        return fused,embedding,weights

class MorphoAgroNet(nn.Module):
    def __init__(self,num_classes,pretrained=True,channels=256,embedding_dim=256,geometry_dim=128,
                 use_mlge=True,use_ggab=True,use_hmf=True):
        super().__init__()
        try: back=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        except Exception as e:
            if pretrained:raise RuntimeError('Could not acquire ResNet-50 ImageNet weights. Download weights, or use --no-pretrained for smoke testing.') from e
            raise
        self.stem=nn.Sequential(back.conv1,back.bn1,back.relu,back.maxpool)
        self.layer1=back.layer1;self.layer2=back.layer2;self.layer3=back.layer3;self.layer4=back.layer4
        self.lesion=LesionEstimator(256)
        self.mlge=MLGE(geometry_dim)
        self.projections=nn.ModuleList([nn.Sequential(nn.Conv2d(c,channels,1,bias=False),nn.BatchNorm2d(channels)) for c in (512,1024,2048)])
        self.attention=nn.ModuleList([GGAB(channels,geometry_dim) for _ in range(3)])
        self.fusion=HMF(channels,embedding_dim)
        self.simple_fuse=nn.Conv2d(channels*3,channels,1)
        self.simple_embed=nn.Linear(channels,embedding_dim)
        self.norm_embedding=nn.Linear(embedding_dim,embedding_dim)
        self.classifier=nn.Sequential(nn.Dropout(.30),nn.Linear(embedding_dim,num_classes))
        self.use_mlge=use_mlge;self.use_ggab=use_ggab;self.use_hmf=use_hmf
        self.geo_dim=geometry_dim
    def freeze_lower(self,freeze=True):
        for layer in [self.stem,self.layer1,self.layer2]:
            for p in layer.parameters():p.requires_grad=not freeze
    def forward(self,x,return_details=False):
        stem=self.stem(x);a=self.layer1(stem);b=self.layer2(a);c=self.layer3(b);d=self.layer4(c)
        soft=self.lesion(a)
        descriptor,binary=batch_exact_geometry(soft)
        if self.use_mlge: geometry,scale_weights=self.mlge(soft,descriptor)
        else:
            geometry=x.new_zeros((x.shape[0],self.geo_dim));scale_weights=x.new_full((x.shape[0],3),1/3)
        feats=[proj(f) for proj,f in zip(self.projections,(b,c,d))]
        if self.use_ggab:
            pairs=[block(f,geometry) for block,f in zip(self.attention,feats)]
            refined,attentions=[p[0] for p in pairs],[p[1] for p in pairs]
        else:
            refined=feats;attentions=[torch.ones_like(f[:,:1]) for f in feats]
        if self.use_hmf:fused,emb,hweights=self.fusion(refined)
        else:
            size=refined[0].shape[-2:]
            maps=[F.interpolate(t,size=size,mode='bilinear',align_corners=False) for t in refined]
            fused=self.simple_fuse(torch.cat(maps,dim=1))
            emb=self.simple_embed(F.adaptive_avg_pool2d(fused,1).flatten(1))
            hweights=x.new_full((x.shape[0],3),1/3)
        normalized=F.normalize(self.norm_embedding(emb),dim=1)
        logits=self.classifier(emb)
        output={'logits':logits,'embedding':emb,'normalized_embedding':normalized,'soft_lesion':soft,
                'binary_lesion':binary,'geometry_descriptor':descriptor,'geometry_weights':scale_weights,
                'hierarchy_weights':hweights,'attention_maps':attentions,'fused_map':fused}
        return output if return_details else logits
