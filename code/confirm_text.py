"""Three-seed OOF confirmation for the selected lexical residual model."""
import json
from pathlib import Path
import numpy as np
from train import labels,metrics
from common import load_inputs
from features import build_features
from text_method import bags,forward_text
from train_text import fit
b,a=load_inputs('train');y,h=labels(b,'train');x=build_features(b,a,'context');ids,values=bags(b);folds=np.empty(len(y),int);rng=np.random.default_rng(20261027);strata=y.sum(1).astype(int)*2+h.astype(int)
for v in np.unique(strata):
 ix=rng.permutation(np.flatnonzero(strata==v));folds[ix]=np.arange(len(ix))%3
winner=json.loads(Path('runs_text/winner.json').read_text());preds=[]
for seed in [17,29,43]:
 s=np.zeros_like(y)
 for f in range(3):
  tr,va=folds!=f,folds==f
  def callback(e,asset):s[va]=forward_text(x[va],ids[va],values[va],asset)
  fit(x[tr],ids[tr],values[tr],y[tr],winner['config'],seed,[winner['epochs']],callback)
 preds.append(s);print(seed,metrics(y,h,s.argmax(1))['hard_accuracy'],flush=True)
s=np.mean(preds,0);np.savez('runs_text/confirmed_oof.npz',scores=s);Path('runs_text/confirmed_oof.json').write_text(json.dumps(metrics(y,h,s.argmax(1)),indent=2))
