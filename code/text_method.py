"""Supervised hashed lexical residual; numeric literals are normalized, no vocabulary or pretrained model."""
import re,zlib,json
from pathlib import Path
from collections import Counter
import numpy as np
from features import build_features,normalize,describe
TOKEN=re.compile(r'[a-z]+|NUM|[+*/=<>%-]')
def bags(bank,dim=16384,limit=512):
 ids=np.zeros((len(bank),4,limit),np.int32);values=np.zeros_like(ids,dtype=np.float32)
 for n,q in enumerate(bank):
  qwords=set(re.findall('[a-z]+',q['question'].lower()))
  for i,c in enumerate(q['candidates']):
   t=c['text'].lower();t=re.sub(r'[-+]?\d[\d,.]*(?:/\d+)?',' NUM ',t);words=TOKEN.findall(t)
   ts=['u:'+w for w in words]+['b:'+a+'_'+b for a,b in zip(words,words[1:])]+['q:'+w for w in words if w in qwords]
   counts=Counter(zlib.crc32(s.encode())%dim for s in ts)
   top=counts.most_common(limit);norm=np.sqrt(sum(v for _,v in top)) or 1
   for j,(key,value) in enumerate(top):ids[n,i,j]=key;values[n,i,j]=np.sqrt(value)/norm
 return ids,values

def forward_text(x,ids,values,a):
 x=normalize(x,a['mean'],a['std']);h=np.maximum(x@a['w1']+a['b1'],0)
 return (h@a['w2']+a['b2']).squeeze(-1)+(a['text_weight'][ids]*values).sum(-1)

def score(bank,arrays,checkpoint):
 checkpoint=Path(checkpoint);cfg=json.loads((checkpoint/'config.json').read_text());x=build_features(bank,arrays,'context');ids,values=bags(bank);scores=[]
 for name in cfg['members']:
  with np.load(checkpoint/name,allow_pickle=False) as a:scores.append(forward_text(x,ids,values,a))
 ans,_=describe(bank);valid=np.array([[v is not None for v in row] for row in ans]);return np.where(valid,np.mean(scores,0),-1e6)
def predict(bank,arrays,checkpoint):return score(bank,arrays,checkpoint).argmax(1).tolist()
