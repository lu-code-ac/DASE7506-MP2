"""Fixed random Fourier features plus a learned linear score; no extra pretrained weights."""
import json
from pathlib import Path
import numpy as np
from features import normalize,build_features
from v2_features import pooled_scores
from method_v2 import normalize_scores
from features import describe

def expand(x,asset):
 x=normalize(x,asset['mean'],asset['std']);return np.concatenate([x,np.cos(x@asset['projection']+asset['phase'])],-1)
def scores(bank,arrays,checkpoint):
 checkpoint=Path(checkpoint);config=json.loads((checkpoint/'config.json').read_text());x=build_features(bank,arrays,'context');raw=[]
 for name in config['members']:
  with np.load(checkpoint/name,allow_pickle=False) as a:raw.append((expand(x,a)@a['weight']+a['bias']).squeeze(-1))
 return pooled_scores(np.mean(raw,0),bank,config['pool'])
def predict(bank,arrays,checkpoint):
 s=scores(bank,arrays,checkpoint);answers,_=describe(bank);valid=np.array([[a is not None for a in row] for row in answers]);return np.where(valid,s,-1e6).argmax(1).tolist()
