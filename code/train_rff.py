import json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from train import labels,metrics
from common import load_inputs
from features import build_features,normalize,normalize_fit,describe
from v2_features import pooled_scores
from rff_method import expand

def fit(x,y,c,seed,epochs,callback):
 rng=np.random.default_rng(314159);torch.manual_seed(seed);shuffle=np.random.default_rng(seed);use=(y.sum(1)<4) if c['loss']=='pair' else np.ones(len(y),bool);x,y=x[use],y[use]
 mean,std=normalize_fit(x);dim=x.shape[-1];asset={'mean':mean,'std':std,'projection':rng.normal(0,np.sqrt(2*c['gamma']/dim),(dim,512)).astype(np.float32),'phase':rng.uniform(0,2*np.pi,512).astype(np.float32)}
 nx=torch.from_numpy(normalize(x,mean,std)).cuda();projection=torch.from_numpy(asset['projection']).cuda();phase=torch.from_numpy(asset['phase']).cuda();tx=torch.cat([nx,torch.cos(nx@projection+phase)],-1);ty=torch.from_numpy(y).cuda();model=nn.Linear(dim+512,1).cuda();opt=torch.optim.Adam(model.parameters(),lr=.003)
 for e in range(1,max(epochs)+1):
  for ids in np.array_split(shuffle.permutation(len(x)),int(np.ceil(len(x)/256))):
   s=model(tx[ids]).squeeze(-1);yy=ty[ids]
   if c['loss']=='bce':loss=torch.nn.functional.binary_cross_entropy_with_logits(s,yy)
   else:
    mask=yy[:,:,None]*(1-yy[:,None,:]);loss=((torch.nn.functional.softplus(s[:,None,:]-s[:,:,None])*mask).sum((1,2))/mask.sum((1,2))).mean()
   loss=loss+c['l2']*model.weight.square().sum();opt.zero_grad();loss.backward();opt.step()
  if e in epochs:
   asset['weight']=model.weight.detach().cpu().numpy().T.copy();asset['bias']=model.bias.detach().cpu().numpy().copy();callback(e,asset)

def main():
 out=Path('runs_rff');out.mkdir(exist_ok=True);b,a=load_inputs('train');y,h=labels(b,'train');x=build_features(b,a,'context');answers,_=describe(b);valid=np.array([[v is not None for v in row] for row in answers]);folds=np.empty(len(y),int);rng=np.random.default_rng(20261027)
 strata=y.sum(1).astype(int)*2+h.astype(int)
 for v in np.unique(strata):
  ids=rng.permutation(np.flatnonzero(strata==v));folds[ids]=np.arange(len(ids))%3
 configs=[dict(gamma=g,l2=l2,loss='bce') for g in [.25,1,4] for l2 in [.001,.01]];results=[];epochs=[10,25,60];(out/'protocol.json').write_text(json.dumps(dict(configs=configs,epochs=epochs,seed=17,projection_seed=314159,test_use='none'),indent=2))
 for ci,c in enumerate(configs):
  preds={e:np.zeros_like(y) for e in epochs}
  for f in range(3):
   tr,va=folds!=f,folds==f
   def callback(e,asset):preds[e][va]=(expand(x[va],asset)@asset['weight']+asset['bias']).squeeze(-1)
   fit(x[tr],y[tr],c,17,epochs,callback)
  for e,s in preds.items():
   for pool in ['none','mean','logmeanexp']:
    ss=pooled_scores(s,b,pool);choice=np.where(valid,ss,-1e6).argmax(1);results.append(dict(config_index=ci,config=c,epochs=e,pool=pool,oof=metrics(y,h,choice)))
  np.savez(out/f'oof_{ci}.npz',**{f'e{e}':s for e,s in preds.items()});(out/'search_results.json').write_text(json.dumps(results,indent=2));print(json.dumps(max((r for r in results if r['config_index']==ci),key=lambda r:r['oof']['hard_correct'])),flush=True)
 winner=max(results,key=lambda r:r['oof']['hard_correct']);(out/'winner.json').write_text(json.dumps(winner,indent=2));print('WINNER '+json.dumps(winner),flush=True)
 bundle=out/'model_bundle';bundle.mkdir(exist_ok=True)
 # The projection is identical for all seeds, so export one projection and average linear heads.
 assets=[]
 for seed in [17,29,43]:
  def callback(e,asset):assets.append(dict(asset))
  fit(x,y,winner['config'],seed,[winner['epochs']],callback)
 merged=dict(assets[0]);merged['weight']=np.mean([a['weight'] for a in assets],0);merged['bias']=np.mean([a['bias'] for a in assets],0);np.savez(bundle/'model.npz',**merged);(bundle/'config.json').write_text(json.dumps(dict(winner,members=['model.npz']),indent=2))
 from rff_method import predict
 vb,va=load_inputs('validation');vy,vh=labels(vb,'validation');choices=predict(vb,va,bundle);report=metrics(vy,vh,choices);(out/'validation_summary.json').write_text(json.dumps(report,indent=2));print('VALIDATION '+json.dumps(report),flush=True)
if __name__=='__main__':main()
