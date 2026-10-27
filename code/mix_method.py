"""Fuse representation ranking and a lexical residual using train-OOF score scales."""
import json
from pathlib import Path
import numpy as np
from features import build_features,forward,describe
from text_method import bags,forward_text

def combine(raw,bank,config):
 answers,_=describe(bank);valid=np.array([[v is not None for v in row] for row in answers]);scores=[]
 for s,scale in zip(raw,config['scales']):
  z=s/scale
  if config['mode']=='answer_probability':
   z=np.where(valid,z/config['temperature'],-1e6);p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
   same=np.array([[[a is not None and a==b for b in row] for a in row] for row in answers]);z=(same*p[:,None,:]).sum(-1)
  scores.append(z)
 w=config['v1_weight'];return np.where(valid,w*scores[0]+(1-w)*scores[1],-1e6)

def score(bank,arrays,checkpoint):
 checkpoint=Path(checkpoint);cfg=json.loads((checkpoint/'config.json').read_text());x=build_features(bank,arrays,'context');ids,values=bags(bank);raw=[]
 for family in ['v1','text']:
  member=[]
  for name in cfg[family+'_members']:
   with np.load(checkpoint/name,allow_pickle=False) as a:member.append(forward(x,a) if family=='v1' else forward_text(x,ids,values,a))
  raw.append(np.mean(member,0))
 return combine(raw,bank,cfg)
def predict(bank,arrays,checkpoint):return score(bank,arrays,checkpoint).argmax(1).tolist()
