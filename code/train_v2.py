"""Bounded, train-only V2 search; no test bank or reference access."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
import json,time,hashlib,argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from train import labels,metrics,diagnostics
from common import load_inputs
from features import normalize_fit,normalize,forward,describe
from v2_features import features,pooled_scores

POOLS=['none','mean','max','logsumexp','logmeanexp']

def fit_v2(x,y,hard,c,seed,checkpoints,callback):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    use=(y.sum(1)<4) if c['loss'] in ('pair','marginal') else np.ones(len(y),bool)
    x,y,hard=x[use],y[use],hard[use]
    mean,std=normalize_fit(x);device=c.get('device','cpu');tx=torch.from_numpy(normalize(x,mean,std)).to(device);ty=torch.from_numpy(y).to(device)
    weights=torch.from_numpy(np.where(hard,c['hard_weight'],1).astype(np.float32)).to(device)
    hidden=c['hidden'];dim=x.shape[-1]
    model=nn.Sequential(nn.Linear(dim,hidden),nn.ReLU(),nn.Dropout(c['dropout']),nn.Linear(hidden,1)) if hidden else nn.Sequential(nn.Linear(dim,1))
    model=model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['wd'])
    for epoch in range(1,max(checkpoints)+1):
        model.train()
        for batch in np.array_split(rng.permutation(len(x)),int(np.ceil(len(x)/256))):
            s=model(tx[batch]).squeeze(-1);yy=ty[batch]
            if c['loss']=='bce':loss=torch.nn.functional.binary_cross_entropy_with_logits(s,yy,reduction='none').mean(1)
            elif c['loss']=='marginal':
                t=c.get('temperature',1.0);z=s/t
                loss=torch.logsumexp(z,dim=1)-torch.logsumexp(z.masked_fill(yy==0,float('-inf')),dim=1)
            elif c['loss']=='list':
                # Uniform distribution over correct candidates, rather than one arbitrary positive.
                target=yy/yy.sum(1,keepdim=True);loss=-(target*torch.log_softmax(s,dim=1)).sum(1)
            else:
                mask=yy[:,:,None]*(1-yy[:,None,:]);loss=(torch.nn.functional.softplus(s[:,None,:]-s[:,:,None])*mask).sum((1,2))/mask.sum((1,2)).clamp(min=1)
            loss=(loss*weights[batch]).sum()/weights[batch].sum()
            if c.get('l2',0):loss=loss+c['l2']*sum(p.square().sum() for p in model.parameters() if p.ndim>1)
            opt.zero_grad();loss.backward();opt.step()
        if epoch in checkpoints:
            asset={'mean':mean,'std':std}
            if hidden:asset.update(w1=model[0].weight.detach().cpu().numpy().T.copy(),b1=model[0].bias.detach().cpu().numpy().copy(),w2=model[3].weight.detach().cpu().numpy().T.copy(),b2=model[3].bias.detach().cpu().numpy().copy())
            else:asset.update(w2=model[0].weight.detach().cpu().numpy().T.copy(),b2=model[0].bias.detach().cpu().numpy().copy())
            callback(epoch,asset)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('runs_v2'));p.add_argument('--configs',type=Path);args=p.parse_args();out=args.output;out.mkdir(exist_ok=True)
    bank,arrays=load_inputs('train');y,hard=labels(bank,'train');folds=np.empty(len(y),int);rng=np.random.default_rng(20261027)
    strata=y.sum(1).astype(int)*2+hard.astype(int)
    for v in np.unique(strata):
        ids=rng.permutation(np.flatnonzero(strata==v));folds[ids]=np.arange(len(ids))%3
    configs=[]
    for mode,loss,hidden,hw in [('embedding','bce',0,1),('context','bce',0,1),('context','bce',64,1),('context','bce',128,2),('context','list',64,1),('context','pair',64,1),('group','bce',64,1),('group','list',64,1),('group','pair',64,1),('hybrid','bce',64,1)]:
        configs.append(dict(mode=mode,loss=loss,hidden=hidden,hard_weight=hw,lr=.001 if hidden else .003,wd=.1,dropout=.35))
    # A fixed, bounded grid; all selections use training-only OOF.
    if args.configs: configs=json.loads(args.configs.read_text())
    checkpoints=[10,25,60,120];records=[];start=time.perf_counter()
    (out/'search_protocol.json').write_text(json.dumps({'configs':configs,'epochs':checkpoints,'pools':POOLS,'fold_seed':20261027,'cv_seed':17,'criterion':'training hard OOF accuracy','test_use':'none; earlier V1 test result was already observed'},indent=2))
    best=None
    for ci,c in enumerate(configs):
        tick=time.perf_counter();x=features(bank,arrays,c['mode']);preds={e:np.zeros_like(y) for e in checkpoints}
        for fold in range(3):
            tr,va=folds!=fold,folds==fold
            def callback(epoch,asset):preds[epoch][va]=forward(x[va],asset)
            fit_v2(x[tr],y[tr],hard[tr],c,17,checkpoints,callback)
        np.savez(out/f'oof_{ci}.npz',**{f'e{e}':s for e,s in preds.items()})
        local=[]
        for epoch,scores in preds.items():
            for pool in POOLS:
                s=pooled_scores(scores,bank,pool);choices=s.argmax(1);m=metrics(y,hard,choices)
                rec={'config_index':ci,'config':c,'epochs':epoch,'pool':pool,'oof':m,'fold_hard_accuracy':[metrics(y[folds==f],hard[folds==f],choices[folds==f])['hard_accuracy'] for f in range(3)]}
                records.append(rec);local.append(rec)
                if best is None or m['hard_correct']>best['oof']['hard_correct']:best=rec
        top=max(local,key=lambda r:r['oof']['hard_correct'])
        print(json.dumps({'completed':ci+1,'seconds':time.perf_counter()-tick,'best_for_config':top,'global_best_hard':best['oof']['hard_accuracy']}),flush=True)
        (out/'search_results.json').write_text(json.dumps(records,indent=2))
    (out/'winner.json').write_text(json.dumps(best,indent=2));(out/'search_cost.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'fit_count':len(configs)*3,'candidate_recipes':len(records)},indent=2))
    print('TRAIN-ONLY WINNER '+json.dumps(best),flush=True)
if __name__=='__main__':main()
