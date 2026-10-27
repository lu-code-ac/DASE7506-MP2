"""Train-only 3-fold selection; validation is read after selecting the recipe."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='4'
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from common import load_inputs,read_jsonl,parse_answer,DATA
from features import build_features,normalize_fit,normalize,describe,majority,forward

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

def labels(bank,split):
    refs=read_jsonl(DATA/f'gsm8k_{split}.jsonl')
    assert [q['id'] for q in bank]==[q['id'] for q in refs]
    answers,hard=describe(bank)
    gold=[parse_answer(q['answer']) for q in refs]
    y=np.array([[a is not None and a==g for a in row] for row,g in zip(answers,gold)],dtype=np.float32)
    assert y.sum(1).min()>=1
    return y,hard

def metrics(y,hard,choices):
    correct=y[np.arange(len(y)),choices].astype(bool)
    return {'n':len(y),'correct':int(correct.sum()),'accuracy':float(correct.mean()*100),'hard_n':int(hard.sum()),'hard_correct':int(correct[hard].sum()),'hard_accuracy':float(correct[hard].mean()*100)}

def fit(x,y,hard,config,seed,epochs):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    useful=(y.sum(1)>0)&(y.sum(1)<4)
    x,y,hard=x[useful],y[useful],hard[useful]
    mean,std=normalize_fit(x); tx=torch.from_numpy(normalize(x,mean,std)); ty=torch.from_numpy(y)
    weight=torch.from_numpy(np.where(hard,config['hard_weight'],1).astype(np.float32))
    dim=x.shape[-1]; hidden=config['hidden']
    model=nn.Sequential(nn.Linear(dim,hidden),nn.ReLU(),nn.Dropout(0.2),nn.Linear(hidden,1)) if hidden else nn.Sequential(nn.Linear(dim,1))
    opt=torch.optim.AdamW(model.parameters(),lr=0.001 if hidden else 0.003,weight_decay=0.03)
    model.train()
    for epoch in range(epochs):
        for batch in np.array_split(rng.permutation(len(x)),max(1,int(np.ceil(len(x)/256)))):
            s=model(tx[batch]).squeeze(-1); yy=ty[batch]
            mask=yy[:,:,None]*(1-yy[:,None,:])
            loss=(torch.nn.functional.softplus(s[:,None,:]-s[:,:,None])*mask).sum((1,2))/mask.sum((1,2)).clamp(min=1)
            loss=(loss*weight[batch]).sum()/weight[batch].sum()
            opt.zero_grad(); loss.backward(); opt.step()
    asset={'mean':mean,'std':std}
    if hidden:
        asset.update(w1=model[0].weight.detach().numpy().T.copy(),b1=model[0].bias.detach().numpy().copy(),w2=model[3].weight.detach().numpy().T.copy(),b2=model[3].bias.detach().numpy().copy())
    else: asset.update(w2=model[0].weight.detach().numpy().T.copy(),b2=model[0].bias.detach().numpy().copy())
    return asset

def diagnostics(bank,y,hard,scores):
    choices=scores.argmax(1); base=np.array(majority(bank)); idx=np.arange(len(y)); correct=y[idx,choices].astype(bool); bc=y[idx,base].astype(bool)
    groups={}; answers,_=describe(bank)
    from collections import Counter
    for i,row in enumerate(answers):
        pattern='-'.join(map(str,sorted(Counter(a for a in row if a is not None).values(),reverse=True)))
        groups.setdefault(pattern,[]).append(i)
    group_metrics={p:metrics(y[ids],hard[ids],choices[ids]) for p,ids in groups.items() if hard[ids].any()}
    margin=np.sort(scores,axis=1)[:,-1]-np.sort(scores,axis=1)[:,-2]
    uncertainty=[]
    for ids in np.array_split(np.argsort(margin[hard]),3):
        hidx=np.flatnonzero(hard)[ids]
        uncertainty.append({'n':len(hidx),'accuracy':float(correct[hidx].mean()*100),'mean_margin':float(margin[hidx].mean())})
    return {'groups':group_metrics,'hard_fixed_majority':int((correct&~bc&hard).sum()),'hard_broke_majority':int((~correct&bc&hard).sum()),'hard_majority_wrong_accuracy':float(correct[hard&~bc].mean()*100),'hard_margin_bins_low_to_high':uncertainty}

def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=25);p.add_argument('--output',type=Path,default=Path('runs'));args=p.parse_args()
    start=time.perf_counter(); out=args.output;out.mkdir(exist_ok=True,parents=True)
    bank,arrays=load_inputs('train');y,hard=labels(bank,'train')
    configs=[{'name':'embedding_linear','mode':'embedding','hidden':0,'hard_weight':1},
             {'name':'context_linear','mode':'context','hidden':0,'hard_weight':1},
             {'name':'context_mlp','mode':'context','hidden':64,'hard_weight':1},
             {'name':'full_mlp','mode':'full','hidden':64,'hard_weight':1},
             {'name':'no_style_mlp','mode':'no_style','hidden':64,'hard_weight':1},
             {'name':'full_mlp_hard2','mode':'full','hidden':64,'hard_weight':2}]
    # Stratify question-level folds by number of correct candidates and hard flag.
    folds=np.empty(len(y),dtype=int);rng=np.random.default_rng(20261027)
    strata=y.sum(1).astype(int)*2+hard.astype(int)
    for value in np.unique(strata):
        ids=rng.permutation(np.flatnonzero(strata==value));folds[ids]=np.arange(len(ids))%3
    results=[]
    for config in configs:
        tick=time.perf_counter(); x=build_features(bank,arrays,config['mode']);scores=np.zeros_like(y)
        fold_scores=[]
        for fold in range(3):
            tr,va=folds!=fold,folds==fold
            asset=fit(x[tr],y[tr],hard[tr],config,17,args.epochs)
            scores[va]=forward(x[va],asset)
            fold_scores.append(metrics(y[va],hard[va],scores[va].argmax(1))['hard_accuracy'])
        result={'config':config,'oof':metrics(y,hard,scores.argmax(1)),'fold_hard_accuracy':fold_scores,'seconds':time.perf_counter()-tick}
        results.append(result); print(json.dumps(result),flush=True)
        (out/'cv_results.json').write_text(json.dumps(results,indent=2))
    winner=max(results,key=lambda r:r['oof']['hard_accuracy']); config=winner['config']
    # Recipe selected using training-only OOF. No official validation tuning.
    bundle=out/'model_bundle';bundle.mkdir(exist_ok=True)
    x=build_features(bank,arrays,config['mode']);members=[];assets=[]
    for seed in (17,29,43):
        asset=fit(x,y,hard,config,seed,args.epochs);name=f'member_{seed}.npz';np.savez(bundle/name,**asset);members.append(name);assets.append(asset)
    metadata={'mode':config['mode'],'members':members,'recipe':config,'epochs':args.epochs,'seeds':[17,29,43],'selection':'training-only three-fold OOF hard accuracy; fixed 25 epochs by default'}
    (bundle/'config.json').write_text(json.dumps(metadata,indent=2))
    # Only now load official validation inputs and labels.
    vb,va=load_inputs('validation');vy,vhard=labels(vb,'validation');vx=build_features(vb,va,config['mode'])
    member_scores=[forward(vx,a) for a in assets];vs=np.mean(member_scores,axis=0)
    rng=np.random.default_rng(921);perms=np.array([rng.permutation(4) for _ in vb]);pvb=[dict(q,candidates=[q['candidates'][j] for j in order]) for q,order in zip(vb,perms)]
    pva={'embeddings':va['embeddings'][np.arange(len(vb))[:,None],perms]};px=build_features(pvb,pva,config['mode']);ps=np.mean([forward(px,a) for a in assets],axis=0)
    mapped=perms[np.arange(len(vb)),ps.argmax(1)]
    # Treat exact score ties as equivalent; report literal selection consistency separately.
    equivalent=np.isclose(vs[np.arange(len(vb)),mapped],vs.max(1),rtol=1e-5,atol=1e-5)
    report={'train_baseline':metrics(y,hard,np.array(majority(bank))),'validation_baseline':metrics(vy,vhard,np.array(majority(vb))),'selected_recipe':config,'validation':metrics(vy,vhard,vs.argmax(1)),'validation_members':[metrics(vy,vhard,s.argmax(1)) for s in member_scores],'diagnostics':diagnostics(vb,vy,vhard,vs),'permutation_score_equivalence':float(equivalent.mean()),'permutation_literal_choice_agreement':float((mapped==vs.argmax(1)).mean()),'asset_bytes':sum(f.stat().st_size for f in bundle.iterdir()),'training_seconds':time.perf_counter()-start,'environment':{'platform':platform.platform(),'processor':platform.processor(),'numpy':np.__version__,'torch':torch.__version__,'device':'CPU','threads':4},'test_evaluated':False}
    (out/'experiment_summary.json').write_text(json.dumps(report,indent=2));np.savez(out/'validation_diagnostics.npz',scores=vs,labels=vy,hard=vhard,choices=vs.argmax(1))
    print(json.dumps(report,indent=2),flush=True)
    hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in bundle.iterdir()}
    (out/'asset_hashes.json').write_text(json.dumps(hashes,indent=2))
if __name__=='__main__':main()
