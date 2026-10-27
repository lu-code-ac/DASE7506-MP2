"""Train-only bounded search for a DeepSets-style comparator."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from train import labels,metrics
from common import load_inputs
from features import normalize,normalize_fit
from set_method import inputs,relations,forward_set,score

class Comparator(nn.Module):
 def __init__(self,dim,width,drop):
  super().__init__();self.enc=nn.Linear(dim,width);self.head=nn.Linear(width*4+dim-1024,64);self.out=nn.Linear(64,1);self.drop=nn.Dropout(drop)
 def forward(self,x,same):
  h=torch.relu(self.enc(x));mean=h.mean(1,keepdim=True).expand_as(h);mx=h.max(1,keepdim=True).values.expand_as(h);agree=same@h
  z=torch.cat([h,mean,mx,agree,x[:,:,1024:]],-1);z=self.drop(torch.relu(self.head(z)));return self.out(z).squeeze(-1)

def fit(x,rel,y,hard,c,seed,epochs,callback):
 torch.manual_seed(seed);rng=np.random.default_rng(seed);device='cuda';mean,std=normalize_fit(x)
 tx=torch.from_numpy(normalize(x,mean,std)).to(device);tr=torch.from_numpy(rel).to(device);ty=torch.from_numpy(y).to(device)
 model=Comparator(x.shape[-1],c['width'],c['drop']).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.1)
 for epoch in range(1,max(epochs)+1):
  model.train()
  for ids in np.array_split(rng.permutation(len(x)),int(np.ceil(len(x)/256))):
   s=model(tx[ids],tr[ids]);yy=ty[ids];mask=yy[:,:,None]*(1-yy[:,None,:]);has=mask.sum((1,2))>0
   pair=(torch.nn.functional.softplus(s[:,None,:]-s[:,:,None])*mask).sum((1,2))/mask.sum((1,2)).clamp(min=1)
   bce=torch.nn.functional.binary_cross_entropy_with_logits(s,yy,reduction='none').mean(1)
   if c['loss']=='pair':loss=pair[has].mean()
   elif c['loss']=='bce':loss=bce.mean()
   else:loss=pair[has].mean()+.3*bce.mean()
   opt.zero_grad();loss.backward();opt.step()
  if epoch in epochs:
   asset={'mean':mean,'std':std}
   for key,layer in [('enc',model.enc),('head',model.head),('out',model.out)]:asset[key+'_w']=layer.weight.detach().cpu().numpy().T.copy();asset[key+'_b']=layer.bias.detach().cpu().numpy().copy()
   # Exported NumPy computation must agree with torch in evaluation mode.
   model.eval()
   with torch.no_grad():expected=model(tx[:8],tr[:8]).cpu().numpy()
   actual=forward_set(x[:8],rel[:8],asset);assert np.allclose(actual,expected,atol=2e-4,rtol=2e-4)
   callback(epoch,asset)

def main():
 start=time.perf_counter();out=Path('runs_set');out.mkdir(exist_ok=True);bank,arrays=load_inputs('train');y,hard=labels(bank,'train');rel,valid=relations(bank);folds=np.empty(len(y),int);rng=np.random.default_rng(20261027)
 strata=y.sum(1).astype(int)*2+hard.astype(int)
 for v in np.unique(strata):
  ids=rng.permutation(np.flatnonzero(strata==v));folds[ids]=np.arange(len(ids))%3
 configs=[dict(rich=rich,width=64,drop=.4,loss=loss) for rich in [False,True] for loss in ['pair','bce','mixed']]
 epochs=[10,25,60,120];(out/'protocol.json').write_text(json.dumps({'configs':configs,'epochs':epochs,'seed':17,'fold_seed':20261027,'test_use':'none'},indent=2));results=[];cache={}
 for ci,c in enumerate(configs):
  if c['rich'] not in cache:cache[c['rich']]=inputs(bank,arrays,c['rich'])
  x=cache[c['rich']];preds={e:np.zeros_like(y) for e in epochs}
  for f in range(3):
   tr,va=folds!=f,folds==f
   def callback(epoch,asset):preds[epoch][va]=forward_set(x[va],rel[va],asset)
   fit(x[tr],rel[tr],y[tr],hard[tr],c,17,epochs,callback)
  for e,s in preds.items():
   choices=np.where(valid,s,-1e6).argmax(1);m=metrics(y,hard,choices);results.append({'config_index':ci,'config':c,'epochs':e,'oof':m})
  np.savez(out/f'oof_{ci}.npz',**{f'e{e}':s for e,s in preds.items()});(out/'search_results.json').write_text(json.dumps(results,indent=2));print(json.dumps(max((r for r in results if r['config_index']==ci),key=lambda r:r['oof']['hard_correct'])),flush=True)
 winner=max(results,key=lambda r:r['oof']['hard_correct']);(out/'winner.json').write_text(json.dumps(winner,indent=2));print('WINNER '+json.dumps(winner),flush=True)
 c=winner['config'];x=cache[c['rich']];bundle=out/'model_bundle';bundle.mkdir(exist_ok=True);members=[]
 for seed in [17,29,43]:
  name=f'seed{seed}.npz'
  def callback(epoch,asset):np.savez(bundle/name,**asset)
  fit(x,rel,y,hard,c,seed,[winner['epochs']],callback);members.append(name)
 config=dict(c,members=members,epochs=winner['epochs'],seeds=[17,29,43]);(bundle/'config.json').write_text(json.dumps(config,indent=2))
 vb,va=load_inputs('validation');vy,vh=labels(vb,'validation');vs=score(vb,va,bundle);record={'validation':metrics(vy,vh,vs.argmax(1)),'asset_bytes':sum(p.stat().st_size for p in bundle.iterdir()),'seconds':time.perf_counter()-start};(out/'validation_summary.json').write_text(json.dumps(record,indent=2));np.savez(out/'validation_scores.npz',scores=vs);print('VALIDATION '+json.dumps(record),flush=True)
if __name__=='__main__':main()
