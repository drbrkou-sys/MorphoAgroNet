"""Semantically compatible cross-dataset transfer mappings. Manual check required for source label variants."""
from ..data.dataset import canonical

SHARED={
 'pepper_bacterial_spot':('pepper_bell_bacterial_spot','bell_pepper_leaf_spot',None),
 'pepper_healthy':('pepper_bell_healthy','bell_pepper_leaf',None),
 'potato_early_blight':('potato_early_blight','potato_leaf_early_blight',None),
 'potato_late_blight':('potato_late_blight','potato_leaf_late_blight',None),
 'tomato_bacterial_spot':('tomato_bacterial_spot','tomato_leaf_bacterial_spot',None),
 'tomato_early_blight':('tomato_early_blight','tomato_early_blight_leaf',None),
 'tomato_late_blight':('tomato_late_blight','tomato_leaf_late_blight',None),
 'tomato_leaf_mould':('tomato_leaf_mould','tomato_mold_leaf',None),
 'tomato_septoria':('tomato_septoria_leaf_spot','tomato_septoria_leaf_spot',None),
 'tomato_yellow_virus':('tomato_yellow_leaf_curl_virus','tomato_leaf_yellow_virus',None),
 'tomato_mosaic':('tomato_mosaic_virus','tomato_leaf_mosaic_virus',None),
 'tomato_healthy':('tomato_healthy','tomato_leaf',None),
 'apple_scab':('apple_scab','apple_scab_leaf','scab'),
 'apple_rust':('apple_cedar_apple_rust','apple_rust_leaf','rust'),
 'apple_healthy':('apple_healthy','apple_leaf','healthy')}
KINDS=['plantvillage','plantdoc','pathology2020']

def mappings(source_names,target_names,source_kind,target_kind):
    i=KINDS.index(source_kind);j=KINDS.index(target_kind)
    src={canonical(s):n for n,s in enumerate(source_names)}
    tar={canonical(s):n for n,s in enumerate(target_names)}
    pairs=[]
    for shared,vs in SHARED.items():
        if vs[i] in src and vs[j] in tar:pairs.append((src[vs[i]],tar[vs[j]],shared))
    if len(pairs)<2:raise ValueError('At least two harmonized classes required; check dataset labels')
    return pairs
