"""Conservative arithmetic and answer-question numerical relations, no labels."""
import re
from fractions import Fraction
import numpy as np
from common import parse_answer
from features import calculate

NUMBER=re.compile(r'(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?')
SPAN=re.compile(r'[\d.()+*/\s-]+')

def clean_math(text):
    text=text.replace('\u00d7','*').replace('\u00f7','/').replace('\u2212','-').replace('\\times','*').replace('\\cdot','*').replace('\\div','/')
    text=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',text)
    text=re.sub(r'\\(?:left|right|,|!|;|quad|qquad)\b','',text)
    for _ in range(3):text=re.sub(r'\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}',r'((\1)/(\2))',text)
    return text.replace('$','').replace('\\(','').replace('\\)','').replace('\\[','').replace('\\]','')

def equations(text):
    normalized=clean_math(text);checked=bad=rounded=0;last=None
    for line in normalized.split('\n'):
        parts=line.split('=')
        for left,right in zip(parts,parts[1:]):
            ls=list(SPAN.finditer(left));rs=list(SPAN.finditer(right))
            if not ls or not rs:continue
            lmatch=ls[-1];rmatch=rs[0]
            # Require arithmetic directly adjacent to the equality (allow whitespace).
            if left[lmatch.end():].strip() or right[:rmatch.start()].strip():continue
            lhs,rhs=lmatch.group().strip(),rmatch.group().strip()
            if not lhs or not rhs or not any(op in lhs for op in '+-*/'):continue
            try:
                a,b=calculate(lhs),calculate(rhs)
                if not np.isfinite(a) or not np.isfinite(b) or abs(a)>1e14 or abs(b)>1e14:continue
                # A printed decimal can be rounded to its last displayed digit.
                simple=re.fullmatch(r'[-+]?\d+(?:\.(\d+))?',rhs)
                tolerance=.50001*10**(-len(simple[1])) if simple and simple[1] else 1e-6
                ok=abs(a-b)<=max(tolerance,1e-8*max(abs(a),abs(b),1))
                checked+=1;bad+=int(not ok);last=b
            except (ValueError,SyntaxError,ZeroDivisionError,OverflowError):pass
    return checked,bad,last

def numbers(text):
    out=[]
    for m in NUMBER.finditer(text):
        try:
            v=float(Fraction(m.group().replace(',','')))
            if np.isfinite(v):out.append(v)
        except (ValueError,ZeroDivisionError,OverflowError):pass
    return out

def rich_features(bank):
    result=[]
    for q in bank:
        qnums=numbers(q['question']);qn=np.array(qnums or [0.],dtype=np.float64);qabs=np.abs(qn);qmax=max(1,float(qabs.max()));qsum=max(1,float(qabs.sum()))
        row=[]
        for c in q['candidates']:
            text=c['text'];parsed=parse_answer(text);ans=float(Fraction(parsed)) if parsed is not None else 0.
            checked,bad,last=equations(text);tn=numbers(text)
            f=[np.log1p(checked),np.log1p(bad),bad/max(checked,1),float(checked>0),float(last is not None and abs(last-ans)<=1e-6*max(1,abs(ans))),
               float(ans<0),float(ans==0),float(abs(ans-round(ans))<1e-8),np.log1p(abs(ans)),np.log1p(abs(ans)/qmax),np.log1p(abs(ans)/qsum),
               float(any(abs(ans-v)<1e-6 for v in qn)),float(any(abs(ans-v)<1e-6 for v in tn[:-1])),
               np.mean([any(abs(v-t)<1e-6 for t in tn) for v in qn]),
               np.log1p(len(tn)),np.log1p(len(qnums)),np.log1p(text.count('\\frac')),np.log1p(text.count('%'))]
            # Simple two-number relationships are generic numerical features, not solved-answer lookups.
            relations=[False]*6
            for i,v in enumerate(qn[:20]):
                for j,w in enumerate(qn[:20]):
                    if i==j:continue
                    vals=[v+w,v-w,v*w,v/w if w else np.inf,(v+w)/2,v*(1-w/100)]
                    for k,z in enumerate(vals):relations[k]|=abs(ans-z)<=1e-6*max(1,abs(ans))
            f += list(map(float,relations));row.append(f)
        result.append(row)
    return np.asarray(result,dtype=np.float32)
