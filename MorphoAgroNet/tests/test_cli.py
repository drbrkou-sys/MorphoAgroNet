from morphoagronet.models.baselines import build_baseline

def test_baseline_resnet():
    m=build_baseline('resnet50',3,pretrained=False)
    assert m.fc.out_features==3
