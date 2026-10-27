import json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from train import labels,metrics
from common import load_inputs
from features import build_features,normalize,normalize_fit,describe
from text_method import bags,forward_text,score

class Model(nn.Module):
 def __init__(self,dim):
  super().__init__();self.hidden=nn.Linear(dim,64);self.out=nn.Linear(64,1);self.text=nn.Embedding(16384,1);nn.init.zeros_(self.text.weight);self.drop=nn.Dropout(.35)
 def forward(self,x,ids,values):return self.out(self.drop(torch.relu(self.hidden(x)))).squeeze(-1)+(self.text(ids).squeeze(-1)*self.drop(values)).sum(-1)

def fit(x,ids,values,y,c,seed,epochs,callback):
 torch.manual_seed(seed);rng=np.random.default_rng(seed)
 use=y.sum(1)<4 if c['loss']=='pair' else np.ones(len(y),bool);x,ids,values,y=x[use],ids[use],values[use],y[use]
 mean,std=normalize_fit(x);tx=torch.from_numpy(normalize(x,mean,std)).cuda();ti=torch.from_numpy(ids).cuda();tv=torch.from_numpy(values).cuda();ty=torch.from_numpy(y).cuda()
 model=Model(x.shape[-1]).cuda();opt=torch.optim.AdamW([{'params':list(model.hidden.parameters())+list(model.out.parameters()),'lr':.001},{'params':model.text.parameters(),'lr':.01}],weight_decay=.1)
 for e in range(1,max(epochs)+1):
  model.train()
  for ix in np.array_split(rng.permutation(len(x)),int(np.ceil(len(x)/256))):
   s=model(tx[ix],ti[ix],tv[ix]);yy=ty[ix];mask=yy[:,:,None]*(1-yy[:,None,:]);pair=(torch.nn.functional.softplus(s[:,None,:]-s[:,:,None])*mask).sum((1,2))/mask.sum((1,2)).clamp(min=1)
   if c['loss']=='pair':loss=pair.mean()
   else:loss=torch.nn.functional.binary_cross_entropy_with_logits(s,yy)+.3*pair.mean()
   loss=loss+c['l2']*model.text.weight.square().sum();opt.zero_grad();loss.backward();opt.step()
  if e in epochs:
   asset={'mean':mean,'std':std,'w1':model.hidden.weight.detach().cpu().numpy().T.copy(),'b1':model.hidden.bias.detach().cpu().numpy().copy(),'w2':model.out.weight.detach().cpu().numpy().T.copy(),'b2':model.out.bias.detach().cpu().numpy().copy(),'text_weight':model.text.weight.detach().cpu().numpy().ravel().copy()}
   model.eval()
   with torch.no_grad():expected=model(tx[:8],ti[:8],tv[:8]).cpu().numpy()
   assert np.allclose(forward_text(x[:8],ids[:8],values[:8],asset),expected,atol=2e-4,rtol=2e-4)
   callback(e,asset)

def main():
 out=Path('runs_text');out.mkdir(exist_ok=True);start=time.perf_counter();b,a=load_inputs('train');y,h=labels(b,'train');x=build_features(b,a,'context');ids,values=bags(b);ans,_=describe(b);valid=np.array([[v is not None for v in row] for row in ans]);folds=np.empty(len(y),int);rng=np.random.default_rng(20261027)
 strata=y.sum(1).astype(int)*2+h.astype(int)
 for v in np.unique(strata):
  ix=rng.permutation(np.flatnonzero(strata==v));folds[ix]=np.arange(len(ix))%3
 configs=[dict(loss=loss,l2=l2) for loss in ['pair','mixed'] for l2 in [.0001,.001]];epochs=[10,25,60];results=[];(out/'protocol.json').write_text(json.dumps(dict(configs=configs,epochs=epochs,seed=17,fold_seed=20261027,test_use='none'),indent=2))
 for ci,c in enumerate(configs):
  preds={e:np.zeros_like(y) for e in epochs}
  for f in range(3):
   tr,va=folds!=f,folds==f
   def callback(e,asset):preds[e][va]=forward_text(x[va],ids[va],values[va],asset)
   fit(x[tr],ids[tr],values[tr],y[tr],c,17,epochs,callback)
  for e,s in preds.items():results.append(dict(config_index=ci,config=c,epochs=e,oof=metrics(y,h,np.where(valid,s,-1e6).argmax(1))))
  np.savez(out/f'oof_{ci}.npz',**{f'e{e}':s for e,s in preds.items()});(out/'search_results.json').write_text(json.dumps(results,indent=2));print(json.dumps(max((r for r in results if r['config_index']==ci),key=lambda r:r['oof']['hard_correct'])),flush=True)
 winner=max(results,key=lambda r:r['oof']['hard_correct']);(out/'winner.json').write_text(json.dumps(winner,indent=2));bundle=out/'model_bundle';bundle.mkdir(exist_ok=True);members=[]
 for seed in [17,29,43]:
  name=f'seed{seed}.npz'
  def callback(e,asset):np.savez(bundle/name,**asset)
  fit(x,ids,values,y,winner['config'],seed,[winner['epochs']],callback);members.append(name)
 (bundle/'config.json').write_text(json.dumps(dict(winner,members=members),indent=2));vb,va=load_inputs('validation');vy,vh=labels(vb,'validation');vs=score(vb,va,bundle);report=dict(validation=metrics(vy,vh,vs.argmax(1)),seconds=time.perf_counter()-start);(out/'validation_summary.json').write_text(json.dumps(report,indent=2));np.savez(out/'validation_scores.npz',scores=vs);print('VALIDATION '+json.dumps(report),flush=True)
if __name__=='__main__':main()
