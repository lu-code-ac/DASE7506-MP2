"""Confirm three preselected families over three seeds; select with train OOF only."""
import json,itertools,time,hashlib
from pathlib import Path
import numpy as np
from common import load_inputs
from train import labels,metrics,diagnostics
from train_v2 import fit_v2
from v2_features import features,pooled_scores
from features import forward,describe
from method_v2 import normalize_scores,score

def main():
    start=time.perf_counter();out=Path('runs_upgrade');out.mkdir(exist_ok=True)
    plan=json.loads((out/'ensemble_candidates.json').read_text());families=[plan['families'][i] for i in [0,2,3]]
    bank,arrays=load_inputs('train');y,hard=labels(bank,'train');answers,_=describe(bank);valid=np.array([[a is not None for a in row] for row in answers]);folds=np.empty(len(y),int);rng=np.random.default_rng(20261027)
    strata=y.sum(1).astype(int)*2+hard.astype(int)
    for v in np.unique(strata):
        ids=rng.permutation(np.flatnonzero(strata==v));folds[ids]=np.arange(len(ids))%3
    norm_scores=[];confirm=[];cache={}
    for fi,fam in enumerate(families):
        c=dict(fam['config']);c['device']='cuda';fam['config']=c;mode=c['mode']
        if mode not in cache:cache[mode]=features(bank,arrays,mode)
        x=cache[mode];seed_scores=[]
        for seed in [17,29,43]:
            ss=np.zeros_like(y)
            for fold in range(3):
                tr,va=folds!=fold,folds==fold
                def callback(epoch,asset):ss[va]=forward(x[va],asset)
                fit_v2(x[tr],y[tr],hard[tr],c,seed,[fam['epochs']],callback)
            seed_scores.append(ss)
            print(json.dumps({'family':fi,'seed':seed,'hard_accuracy':metrics(y,hard,pooled_scores(ss,bank,fam['pool']).argmax(1))['hard_accuracy']}),flush=True)
        raw=np.mean(seed_scores,0);pooled=pooled_scores(raw,bank,fam['pool']);ns=normalize_scores(pooled,valid);norm_scores.append(ns)
        confirm.append({'family':fam,'individual_seeds':[metrics(y,hard,pooled_scores(s,bank,fam['pool']).argmax(1)) for s in seed_scores],'ensemble_oof':metrics(y,hard,ns.argmax(1))})
        np.savez(out/f'confirmed_oof_{fi}.npz',scores=np.array(seed_scores),norm=ns)
    comparisons=[]
    for n in range(1,4):
        for ids in itertools.combinations(range(3),n):
            s=np.mean([norm_scores[i] for i in ids],0);choice=s.argmax(1)
            comparisons.append({'families':list(ids),'oof':metrics(y,hard,choice),'fold_hard_accuracy':[metrics(y[folds==f],hard[folds==f],choice[folds==f])['hard_accuracy'] for f in range(3)]})
    selected=max(comparisons,key=lambda r:(r['oof']['hard_correct'],-len(r['families'])))
    record={'families':confirm,'comparisons':comparisons,'selected':selected,'validation_accessed':False,'test_accessed':False,'search_fit_count':27,'seconds':time.perf_counter()-start}
    (out/'confirmation.json').write_text(json.dumps(record,indent=2));print('SELECTED '+json.dumps(selected),flush=True)
    bundle=out/'model_bundle';bundle.mkdir(exist_ok=True);final_families=[]
    for fi in selected['families']:
        fam=dict(families[fi]);c=fam['config'];x=cache[c['mode']];names=[]
        for seed in [17,29,43]:
            name=f'family{fi}_seed{seed}.npz'
            def callback(epoch,asset):np.savez(bundle/name,**asset)
            fit_v2(x,y,hard,c,seed,[fam['epochs']],callback);names.append(name)
        fam['members']=names;final_families.append(fam)
    config={'families':final_families,'seeds':[17,29,43],'method':'answer pooling; equal normalized family scores; invalid answers excluded','selection':'train-only OOF; V1 test observed earlier but never used to choose V2 recipes'}
    (bundle/'config.json').write_text(json.dumps(config,indent=2))
    # Validation is used only after train-only selection and final fitting.
    vb,va=load_inputs('validation');vy,vh=labels(vb,'validation');vs=score(vb,va,bundle)
    validation={'metrics':metrics(vy,vh,vs.argmax(1)),'diagnostics':diagnostics(vb,vy,vh,vs),'asset_bytes':sum(p.stat().st_size for p in bundle.iterdir()),'total_confirmation_and_fit_seconds':time.perf_counter()-start}
    (out/'validation_summary.json').write_text(json.dumps(validation,indent=2));np.savez(out/'validation_diagnostics.npz',scores=vs,labels=vy,hard=vh,choices=vs.argmax(1));print('VALIDATION '+json.dumps(validation),flush=True)
    src=['method_v2.py','v2_features.py','rich_features.py','features.py','train_v2.py','confirm_upgrade.py'];files=[Path(s) for s in src]+list(bundle.iterdir())
    freeze={'stage':'V2 frozen after validation report and before any V2 test evaluation','V1_test_previously_observed':True,'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (out/'freeze_manifest.json').write_text(json.dumps(freeze,indent=2))
if __name__=='__main__':main()
