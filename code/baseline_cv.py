"""Reconstruct the V1 three-seed OOF baseline for a matched comparison."""
import json
from pathlib import Path
import numpy as np
from common import load_inputs
from train import labels,metrics,fit
from features import build_features,forward
b,a=load_inputs('train');y,h=labels(b,'train');x=build_features(b,a,'context');folds=np.empty(len(y),int);rng=np.random.default_rng(20261027)
strata=y.sum(1).astype(int)*2+h.astype(int)
for v in np.unique(strata):
 ids=rng.permutation(np.flatnonzero(strata==v));folds[ids]=np.arange(len(ids))%3
scores=[]
for seed in [17,29,43]:
 s=np.zeros_like(y)
 for f in range(3):
  tr,va=folds!=f,folds==f;asset=fit(x[tr],y[tr],h[tr],dict(hidden=64,hard_weight=1),seed,25);s[va]=forward(x[va],asset)
 scores.append(s)
r=metrics(y,h,np.mean(scores,0).argmax(1));Path('runs_upgrade/v1_matched_oof.json').write_text(json.dumps(r,indent=2));np.savez('runs_upgrade/v1_matched_oof.npz',scores=np.mean(scores,0));print(json.dumps(r))
