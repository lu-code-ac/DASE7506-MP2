"""V2 feature variants; reference-free and permutation-equivariant."""
import numpy as np
from features import build_features,describe

def features(bank,arrays,mode):
    if mode.endswith('_rich'):
        from rich_features import rich_features
        return np.concatenate([features(bank,arrays,mode[:-5]),rich_features(bank)],axis=-1)
    if mode in ('embedding','context'):return build_features(bank,arrays,mode)
    x=build_features(bank,arrays,'context');emb=x[:,:,:1024];answers,_=describe(bank)
    means=np.zeros_like(emb);maxima=np.zeros_like(emb);scalar_mean=np.zeros_like(x[:,:,2048:]);scalar_max=np.zeros_like(scalar_mean)
    for n,row in enumerate(answers):
        for i,a in enumerate(row):
            ids=[j for j,b in enumerate(row) if a is not None and a==b] if a is not None else [i]
            means[n,i]=emb[n,ids].mean(0);maxima[n,i]=emb[n,ids].max(0)
            scalar_mean[n,i]=x[n,ids,2048:].mean(0);scalar_max[n,i]=x[n,ids,2048:].max(0)
    if mode=='group':return np.concatenate([means,maxima,means-emb.mean(1,keepdims=True),scalar_mean,scalar_max],-1)
    if mode=='hybrid':return np.concatenate([x,means,means-emb.mean(1,keepdims=True)],-1)
    raise ValueError(mode)

def pooled_scores(scores,bank,pool):
    if pool=='none':return scores
    answers,_=describe(bank);result=scores.copy()
    for n,row in enumerate(answers):
        for a in set(row):
            if a is None:continue
            ids=[j for j,b in enumerate(row) if b==a];s=scores[n,ids]
            if pool=='mean':v=s.mean()
            elif pool=='max':v=s.max()
            elif pool=='logsumexp':v=s.max()+np.log(np.exp(s-s.max()).sum())
            elif pool=='logmeanexp':v=s.max()+np.log(np.exp(s-s.max()).mean())
            else:raise ValueError(pool)
            result[n,ids]=v
    return result
