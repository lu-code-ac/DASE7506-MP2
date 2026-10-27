"""Permutation-equivariant learned comparison of four candidates."""
import json
from pathlib import Path
import numpy as np
from features import build_features,normalize
from rich_features import rich_features
from common import parse_answer

def inputs(bank,arrays,rich):
    emb=np.asarray(arrays['embeddings'],dtype=np.float32)
    scalars=build_features(bank,arrays,'context')[:,:,2048:]
    if rich:scalars=np.concatenate([scalars,rich_features(bank)],-1)
    return np.concatenate([emb,scalars],-1)

def relations(bank):
    answers=[[parse_answer(c['text']) for c in q['candidates']] for q in bank]
    valid=np.array([[a is not None for a in row] for row in answers])
    same=np.array([[[float(a is not None and a==b) for b in row] for a in row] for row in answers],np.float32)
    # Invalid answers do not form a shared group; each gets its own self edge.
    same[:,np.arange(4),np.arange(4)]=1
    same/=same.sum(-1,keepdims=True)
    return same,valid

def forward_set(x,same,asset):
    x=normalize(x,asset['mean'],asset['std'])
    h=np.maximum(x@asset['enc_w']+asset['enc_b'],0)
    mean=np.broadcast_to(h.mean(1,keepdims=True),h.shape)
    maximum=np.broadcast_to(h.max(1,keepdims=True),h.shape)
    agree=same@h
    z=np.concatenate([h,mean,maximum,agree,x[:,:,1024:]],-1)
    z=np.maximum(z@asset['head_w']+asset['head_b'],0)
    return (z@asset['out_w']+asset['out_b']).squeeze(-1)

def score(bank,arrays,checkpoint):
    checkpoint=Path(checkpoint);config=json.loads((checkpoint/'config.json').read_text());x=inputs(bank,arrays,config['rich']);same,valid=relations(bank);ss=[]
    for name in config['members']:
        with np.load(checkpoint/name,allow_pickle=False) as asset:ss.append(forward_set(x,same,asset))
    return np.where(valid,np.mean(ss,0),-1e6)

def predict(bank,arrays,checkpoint):return score(bank,arrays,checkpoint).argmax(1).tolist()
