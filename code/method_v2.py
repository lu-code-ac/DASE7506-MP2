"""V2 NumPy inference with answer pooling and per-question score normalization."""
import json
from pathlib import Path
import numpy as np
from features import forward,describe
from v2_features import features,pooled_scores

def normalize_scores(scores,valid):
    count=np.maximum(valid.sum(1,keepdims=True),1)
    mean=(scores*valid).sum(1,keepdims=True)/count
    std=np.sqrt(((scores-mean)**2*valid).sum(1,keepdims=True)/count)
    out=np.clip((scores-mean)/np.maximum(std,.1),-10,10)
    return np.where(valid,out,-100)

def score(bank,arrays,checkpoint):
    checkpoint=Path(checkpoint);config=json.loads((checkpoint/'config.json').read_text());answers,_=describe(bank)
    valid=np.array([[a is not None for a in row] for row in answers]);cache={};scores=[]
    for family in config['families']:
        mode=family['config']['mode']
        if mode not in cache:cache[mode]=features(bank,arrays,mode)
        raw=[]
        for name in family['members']:
            with np.load(checkpoint/name,allow_pickle=False) as asset:raw.append(forward(cache[mode],asset))
        pooled=pooled_scores(np.mean(raw,0),bank,family['pool']);scores.append(normalize_scores(pooled,valid))
    return np.mean(scores,0)

def predict(bank,arrays,checkpoint):return score(bank,arrays,checkpoint).argmax(1).tolist()
