import numpy as np
from PIL import Image
from morphoagronet.data.dataset import make_manifest,DiseaseDataset
from morphoagronet.data.preprocessing import ImageTransform
from morphoagronet.data.pseudo_masks import pseudo_mask_rgb

def test_discovery_and_split(tmp_path):
    for j in range(3):
        folder=tmp_path/f'label_{j}';folder.mkdir()
        for i in range(15):
            a=np.zeros((24,24,3),dtype=np.uint8)
            a[:,:,1]=70+j*40;a[:,:,0]=i*8
            Image.fromarray(a).save(folder/f'{i}.png')
    manifest=tmp_path/'manifest.csv'
    df=make_manifest(tmp_path,manifest)
    assert len(df)==45
    assert set(df.split)=={'train','val','test'}
    ds=DiseaseDataset(manifest,'train',ImageTransform(False,64))
    x,y=ds[0];assert x.shape==(3,64,64) and isinstance(y,int)

def test_pseudo_mask():
    im=np.full((32,32,3),[10,120,10],dtype=np.uint8)
    im[10:18,10:18]=[110,50,10]
    assert pseudo_mask_rgb(im).shape==(32,32)
    assert pseudo_mask_rgb(im,healthy=True).sum()==0
