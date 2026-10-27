"""Input-only features; unknown arithmetic is not an error."""
import ast
import operator
import re
from collections import Counter
import numpy as np
from common import parse_answer
OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}
NUM=r'[-+]?\d+(?:\.\d+)?'
EQ=re.compile(r'('+NUM+r'(?:\s*[+*/-]\s*'+NUM+r'){1,5})\s*=\s*('+NUM+r')(?![\d.])')
def calculate(expr):
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float): return n.value
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)): return (-1 if isinstance(n.op,ast.USub) else 1)*visit(n.operand)
        if isinstance(n,ast.BinOp) and type(n.op) in OPS: return OPS[type(n.op)](visit(n.left),visit(n.right))
        raise ValueError('unsupported')
    return visit(ast.parse(expr,mode='eval').body)
def arithmetic(text):
    text=text.replace('\u00d7','*').replace('\u00f7','/').replace('\u2212','-')
    good=bad=0
    for line in text.split('\n'):
        m=EQ.fullmatch(line.strip().strip('$').strip())
        if not m: continue
        try:
            lhs,rhs=calculate(m[1]),float(m[2])
            if not np.isfinite(lhs): continue
            ok=abs(lhs-rhs)<=1e-6*max(1,abs(rhs))
            good+=int(ok); bad+=int(not ok)
        except (ValueError,SyntaxError,ZeroDivisionError,OverflowError): pass
    return [np.log1p(good),np.log1p(bad),bad/max(good+bad,1),float(good+bad>0)]
def describe(bank):
    answers=[[parse_answer(c['text']) for c in q['candidates']] for q in bank]
    return answers,np.array([len({a for a in row if a is not None})>=2 for row in answers])
def majority(bank):
    answers,_=describe(bank)
    return [max(range(4),key=lambda i:Counter(x for x in row if x is not None).get(row[i],0)) for row in answers]
def build_features(bank,arrays,mode):
    emb=np.asarray(arrays['embeddings'],dtype=np.float32)
    if mode=='embedding': return emb
    answers,_=describe(bank); all_feats=[]
    for q,row in zip(bank,answers):
        count=Counter(a for a in row if a is not None); votes=sorted(count.values(),reverse=True)+[0,0]
        lengths=[len(c['text']) for c in q['candidates']]; feats=[]
        for c,a,length in zip(q['candidates'],row,lengths):
            f=[float(a is not None),count.get(a,0)/4,len(count)/4,votes[0]/4,(votes[0]-votes[1])/4,row.count(None)/4,float(a is not None and count[a]==votes[0])]
            if mode!='no_style':
                t=c['text']; f += [np.log1p(length),length/max(sum(lengths)/4,1),np.log1p(t.count('\n')),np.log1p(t.count('=')),float('####' in t),float('\\boxed' in t)]
            if mode in ('full','no_style'): f+=arithmetic(c['text'])
            feats.append(f)
        all_feats.append(feats)
    return np.concatenate([emb,emb-emb.mean(axis=1,keepdims=True),np.array(all_feats,dtype=np.float32)],axis=-1)
def normalize_fit(x):
    return x.mean(axis=(0,1)),np.maximum(x.std(axis=(0,1)),0.05)
def normalize(x,mean,std): return np.clip((x-mean)/std,-10,10).astype(np.float32)
def forward(x,asset):
    x=normalize(x,asset['mean'],asset['std'])
    if 'w1' in asset: x=np.maximum(x@asset['w1']+asset['b1'],0)
    return (x@asset['w2']+asset['b2']).squeeze(-1)
