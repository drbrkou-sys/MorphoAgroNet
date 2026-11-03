from pathlib import Path as _ProjectPath
import sys as _project_sys
_project_sys.path.insert(0,str(_ProjectPath(__file__).resolve().parents[1]))
import argparse,json
from morphoagronet.data.dataset import make_manifest
p=argparse.ArgumentParser(description='Construct fixed 70/10/20 split manifests')
p.add_argument('--root',required=True);p.add_argument('--output',required=True)
p.add_argument('--kind',choices=['folder','plantvillage','plantdoc','pathology2020'],default='folder')
p.add_argument('--seed',type=int,default=42)
a=p.parse_args();df=make_manifest(a.root,a.output,a.kind,a.seed)
print(json.dumps({'total':len(df),'classes':df.label.nunique(),'splits':df.split.value_counts().to_dict()},indent=2))
