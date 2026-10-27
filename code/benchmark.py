"""Measure fresh official prediction plus scoring, four threads, without shell timing ambiguity."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['validation','test'],default='validation');args=p.parse_args()
    env=os.environ.copy()
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='4'
    runs=Path('runs');runs.mkdir(exist_ok=True)
    commands=[[sys.executable,'predict.py','--method','method.py','--checkpoint','runs/model_bundle','--split',args.split,'--output',f'runs/{args.split}-predictions.json'],[sys.executable,'score_predictions.py','--split',args.split,'--predictions',f'runs/{args.split}-predictions.json','--output',f'runs/{args.split}-score.json']]
    tick=time.perf_counter();records=[]
    for command in commands:
        start=time.perf_counter();subprocess.run(command,env=env,check=True);records.append({'command':command,'wall_seconds':time.perf_counter()-start})
    result={'split':args.split,'total_wall_seconds':time.perf_counter()-tick,'commands':records,'threads':4,'reference_machine_eligibility':'not verified; local hardware only'}
    (runs/f'{args.split}-timing.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
