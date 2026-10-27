"""Offline NumPy inference. No reference files or torch import."""
import json
from pathlib import Path
import numpy as np
from features import build_features,forward,majority
def predict(bank,arrays,checkpoint):
    if checkpoint is None: return majority(bank)
    checkpoint=Path(checkpoint)
    config=json.loads((checkpoint/'config.json').read_text())
    x=build_features(bank,arrays,config['mode']); scores=[]
    for name in config['members']:
        with np.load(checkpoint/name,allow_pickle=False) as asset: scores.append(forward(x,asset))
    return np.mean(scores,axis=0).argmax(axis=1).tolist()
