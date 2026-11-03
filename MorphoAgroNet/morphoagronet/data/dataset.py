"""Dataset discovery, manifest creation, integrity checks, and image loading."""
import os, re, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

EXT={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}
SEED=42

def canonical(s):
    return re.sub(r'[^a-z0-9]+','_',str(s).lower()).strip('_')

def _hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()

def discover_folder(root):
    """Scan ImageFolder and nested train/test/class directory styles."""
    root=Path(root)
    rows=[]
    for p in sorted(root.rglob('*')):
        if p.suffix.lower() not in EXT:continue
        rel=p.relative_to(root)
        dirs=rel.parts[:-1]
        if not dirs:continue
        skip={'train','test','val','valid','validation','images','dataset','color','rgb'}
        lab=next((d for d in reversed(dirs) if d.lower() not in skip),None)
        if lab is None: continue
        rows.append({'path':str(p.resolve()),'label':lab,'id':str(rel)})
    return pd.DataFrame(rows,columns=['path','label','id'])

def discover_pathology2020(root):
    """Kaggle Plant Pathology 2020 train.csv with one-hot healthy,multiple_diseases,rust,scab."""
    root=Path(root); csv=root/'train.csv'
    if not csv.exists():raise FileNotFoundError(csv)
    df=pd.read_csv(csv)
    labels=['healthy','multiple_diseases','rust','scab']
    if not set(labels).issubset(df):raise ValueError('Plant Pathology train.csv missing class columns')
    rows=[]
    for _,r in df.iterrows():
        values=np.array([r[k] for k in labels],dtype=float)
        if values.sum()!=1:continue
        ident=str(r['image_id'])
        opts=[root/'images'/f'{ident}.jpg',root/f'{ident}.jpg',root/'train_images'/f'{ident}.jpg']
        p=next((x for x in opts if x.exists()),None)
        if p is not None:rows.append({'path':str(p.resolve()),'label':labels[values.argmax()],'id':ident})
    return pd.DataFrame(rows,columns=['path','label','id'])

def discover(root, kind='folder'):
    if kind=='pathology2020':return discover_pathology2020(root)
    df=discover_folder(root)
    if kind=='plantdoc':df=df[df.label.map(canonical)!='tomato_two_spotted_spider_mites_leaf'].copy()
    if kind=='plantvillage':
        crops=('pepper','potato','tomato')
        df=df[df.label.map(lambda s:canonical(s).startswith(crops))].copy()
    return df.reset_index(drop=True)

def make_manifest(root, output, kind='folder', seed=SEED, deduplicate=True):
    df=discover(root,kind)
    if df.empty:raise ValueError(f'No recognized images found in {root}')
    if deduplicate:
        df['sha256']=df.path.map(_hash)
        conflicts=df.groupby('sha256').label.nunique()
        if (conflicts>1).any():raise ValueError('Conflicting labels for identical image bytes; resolve before splitting')
        df=df.drop_duplicates('sha256').reset_index(drop=True)
    counts=df.label.value_counts()
    if (counts<3).any():raise ValueError(f'Classes with fewer than 3 samples: {counts[counts<3].to_dict()}')
    # stratified splits, seed fixed across runs: 70 / 10 / 20
    try:
        tr,hold=train_test_split(np.arange(len(df)),test_size=.30,stratify=df.label,random_state=seed)
        va,te=train_test_split(hold,test_size=2/3,stratify=df.iloc[hold].label,random_state=seed)
    except ValueError as e:raise ValueError(f'Insufficient samples for stratification: {e}') from e
    df['split']='';df.loc[tr,'split']='train';df.loc[va,'split']='val';df.loc[te,'split']='test'
    df['class_id']=df.label.map({s:i for i,s in enumerate(sorted(df.label.unique()))})
    Path(output).parent.mkdir(parents=True,exist_ok=True);df.to_csv(output,index=False)
    return df

class DiseaseDataset(Dataset):
    def __init__(self,manifest,split,transform=None,return_path=False):
        df=pd.read_csv(manifest) if not isinstance(manifest,pd.DataFrame) else manifest.copy()
        self.df=df[df.split==split].reset_index(drop=True)
        if self.df.empty:raise ValueError(f'Empty split {split}')
        self.transform=transform;self.return_path=return_path
    def __len__(self):return len(self.df)
    def __getitem__(self,idx):
        r=self.df.iloc[idx]
        with Image.open(r.path) as im: img=ImageOps.exif_transpose(im).convert('RGB')
        if self.transform:img=self.transform(img)
        else:
            from torchvision.transforms.functional import to_tensor
            img=to_tensor(img)
        if self.return_path:return img,int(r.class_id),str(r.path)
        return img,int(r.class_id)

def class_names(manifest):
    df=pd.read_csv(manifest)
    return df[['label','class_id']].drop_duplicates().sort_values('class_id').label.tolist()
