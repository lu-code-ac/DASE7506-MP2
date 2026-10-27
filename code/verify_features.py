"""Small independent checks for arithmetic semantics and scorer compatibility."""
import numpy as np
from features import arithmetic,calculate,majority,build_features
from common import parse_answer
assert calculate('7*12')==84
assert arithmetic('7 * 12 = 84')[0]>0
assert arithmetic('7 * 12 = 85')[1]>0
assert arithmetic(r'\frac{7}{12} = 0.58')==[0,0,0,0]
assert arithmetic('7 bags * 12 ounces = 85 ounces')==[0,0,0,0]
assert arithmetic('1 / 0 = 0')==[0,0,0,0]
assert parse_answer('#### 1/2')==parse_answer('#### 0.5')
bank=[{'question':'example','candidates':[{'index':i,'text':t} for i,t in enumerate(['#### 1/2','#### 0.5','#### 2','no final'])]}]
assert majority(bank)==[0]
x=build_features(bank,{'embeddings':np.zeros((1,4,1024),np.float32)},'full')
assert x.shape==(1,4,2065) and np.isfinite(x).all()
print('Feature checks passed.')
