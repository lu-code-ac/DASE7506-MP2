"""Select a small fusion grid on training-only three-seed OOF; then evaluate validation."""
import json,shutil,hashlib
from pathlib import Path
import numpy as np
from common import load_inputs
from train import labels,metrics
from features import describe
from mix_method import combine,score

def main():
 b,a=load_inputs('train');y,h=labels(b,'train');answers,_=describe(b);valid=np.array([[v is not None for v in row] for row in answers]);raw=[np.load('runs_upgrade/v1_matched_oof.npz')['scores'],np.load('runs_text/confirmed_oof.npz')['scores']];scales=[]
 for s in raw:
  mean=(s*valid).sum(1,keepdims=True)/valid.sum(1,keepdims=True);scales.append(float(np.sqrt((((s-mean)**2*valid).sum(1)/valid.sum(1))[h].mean())))
 rows=[]
 for mode in ['candidate_score','answer_probability']:
  for w in [.25,.5,.75]:
   for temp in ([1] if mode=='candidate_score' else [.25,.5,1,2]):
    c=dict(mode=mode,v1_weight=w,temperature=temp,scales=scales);s=combine(raw,b,c);rows.append(dict(config=c,oof=metrics(y,h,s.argmax(1))))
 winner=max(rows,key=lambda r:r['oof']['hard_correct']);out=Path('runs_mix');out.mkdir(exist_ok=True);(out/'selection.json').write_text(json.dumps(dict(comparisons=rows,winner=winner,selection='training-only three-seed OOF; no validation labels in selection'),indent=2));print('SELECTED '+json.dumps(winner),flush=True)
 bundle=out/'model_bundle';bundle.mkdir(exist_ok=True);config=winner['config']
 for family,source in [('v1','runs/model_bundle'),('text','runs_text/model_bundle')]:
  names=[]
  for p in sorted(Path(source).glob('*.npz')):
   name=family+'_'+p.name;shutil.copy2(p,bundle/name);names.append(name)
  config[family+'_members']=names
 (bundle/'config.json').write_text(json.dumps(config,indent=2));vb,va=load_inputs('validation');vy,vh=labels(vb,'validation');vs=score(vb,va,bundle);report=dict(validation=metrics(vy,vh,vs.argmax(1)),asset_bytes=sum(p.stat().st_size for p in bundle.iterdir()));(out/'validation_summary.json').write_text(json.dumps(report,indent=2));np.savez(out/'validation_scores.npz',scores=vs);print('VALIDATION '+json.dumps(report),flush=True)
 files=[Path(n) for n in ['mix_method.py','text_method.py','features.py','train_text.py','confirm_text.py','select_mix.py']]+list(bundle.iterdir());(out/'freeze_manifest.json').write_text(json.dumps({'stage':'before any V2 test evaluation','V1_test_observed_previously':True,'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2))
if __name__=='__main__':main()
