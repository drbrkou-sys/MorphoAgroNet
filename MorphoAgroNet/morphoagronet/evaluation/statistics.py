"""Repeated-seed inference and paired image-level statistics."""
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

def compare_runs(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if len(a)!=len(b) or len(a)<2:raise ValueError('Paired arrays with at least 2 runs required')
    d=a-b
    shapiro=float(stats.shapiro(d).pvalue) if len(d)>=3 and len(d)<=5000 else None
    parametric=shapiro is not None and shapiro>=.05
    if parametric:
        test=stats.ttest_rel(a,b);effect=float(d.mean()/(d.std(ddof=1)+1e-12));name='paired_t'
    else:
        test=stats.wilcoxon(d) if np.any(d!=0) else None
        npos=np.sum(d>0);nneg=np.sum(d<0)
        effect=float((npos-nneg)/(npos+nneg)) if npos+nneg else 0.
        name='wilcoxon'
    se=d.std(ddof=1)/np.sqrt(len(d))
    ci=stats.t.interval(.95,len(d)-1,loc=d.mean(),scale=se)
    return {'test':name,'p':float(test.pvalue) if test is not None else 1.,'shapiro_p':shapiro,
            'mean_difference':float(d.mean()),'effect_size':effect,'ci95':[float(ci[0]),float(ci[1])],
            'a_mean':float(a.mean()),'a_sd':float(a.std(ddof=1)),
            'b_mean':float(b.mean()),'b_sd':float(b.std(ddof=1))}

def holm_adjust(pvalues):
    p=np.asarray(pvalues);order=np.argsort(p);adj=np.empty(len(p));last=0.
    for rank,idx in enumerate(order):last=max(last,min(1.,(len(p)-rank)*p[idx]));adj[idx]=last
    return adj.tolist()

def stratified_bootstrap_delta(y,a,b,n=2000,seed=42):
    y=np.asarray(y);a=np.asarray(a);b=np.asarray(b);rng=np.random.default_rng(seed)
    groups=[np.where(y==c)[0] for c in np.unique(y)]
    deltas=[]
    for _ in range(n):
        ids=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups])
        deltas.append(f1_score(y[ids],a[ids],average='macro')-f1_score(y[ids],b[ids],average='macro'))
    return {'mean_delta':float(np.mean(deltas)),'ci95':np.quantile(deltas,[.025,.975]).tolist()}

def mcnemar(y,a,b):
    y=np.asarray(y);a=np.asarray(a);b=np.asarray(b)
    n01=int(np.sum((a==y)&(b!=y)));n10=int(np.sum((a!=y)&(b==y)))
    return {'n01':n01,'n10':n10,'p':float(stats.binomtest(min(n01,n10),n01+n10,.5).pvalue) if n01+n10 else 1.}
