import numpy as np
import torch
from morphoagronet.models.morphoagronet import MorphoAgroNet
from morphoagronet.models.geometry import descriptor_from_mask,batch_exact_geometry
from morphoagronet.training.losses import PrototypeMemory,fgss_loss,lesion_loss,earl_loss
from morphoagronet.evaluation.explainability import explanation,localization_scores
from morphoagronet.evaluation.metrics import classification_metrics
from morphoagronet.evaluation.statistics import compare_runs,holm_adjust

def test_geometry_empty_and_valid():
    a=descriptor_from_mask(np.zeros((32,32),dtype=np.uint8));assert np.isfinite(a).all()
    m=np.zeros((32,32),dtype=np.uint8);m[5:16,6:17]=1
    d=descriptor_from_mask(m);assert d.shape==(8,) and d[5]>0
    soft=torch.rand(2,1,32,32);v,mask=batch_exact_geometry(soft)
    assert v.shape==(2,8) and mask.shape==soft.shape

def test_model_and_gradients():
    torch.set_num_threads(2)
    m=MorphoAgroNet(3,pretrained=False).eval()
    x=torch.randn(2,3,64,64)
    r=m(x,return_details=True)
    assert r['logits'].shape==(2,3)
    assert r['soft_lesion'].shape==(2,1,16,16)
    assert torch.allclose(r['geometry_weights'].sum(1),torch.ones(2),atol=1e-5)
    assert torch.allclose(r['hierarchy_weights'].sum(1),torch.ones(2),atol=1e-5)
    loss=r['logits'].sum();loss.backward()
    assert m.lesion.net[0].weight.grad is not None
    assert torch.isfinite(m.lesion.net[0].weight.grad).all()

def test_losses():
    z=torch.nn.functional.normalize(torch.randn(4,256),dim=1);y=torch.tensor([0,0,1,1])
    mem=PrototypeMemory(3);mem.update(z,y)
    fg,_=fgss_loss(z,y,mem);assert torch.isfinite(fg)
    assert torch.isfinite(lesion_loss(torch.rand(4,1,8,8),torch.rand(4,1,32,32)))
    out={'normalized_embedding':z,'logits':torch.randn(4,3)}
    robust,_=earl_loss(out,out);assert robust.abs()<1e-5

def test_metrics_and_stats():
    r=classification_metrics([0,1,0,1],[0,1,1,1]);assert 0<r['macro_f1']<1
    s=compare_runs([.9,.89,.92,.91,.94],[.88,.87,.9,.89,.9]);assert s['mean_difference']>0
    assert len(holm_adjust([.01,.02,.2]))==3

def test_localization():
    h=np.zeros((16,16));h[3:8,3:8]=1
    r=localization_scores(h,h);assert r['pointing_game']==1
