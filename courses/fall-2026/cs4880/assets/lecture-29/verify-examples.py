"""Verify equations and execute the exact Python blocks in Lecture 29.

Requires numpy, scikit-learn, tensorflow, matplotlib. Use KERAS_HOME to select a
writable cache. MNIST is downloaded if absent. This performs the displayed 30
cancer epochs, 100 sine epochs, and 3 MNIST epochs, not a shortened substitute.
The supplied historical plots remain separate from these measured results.
"""
from pathlib import Path
import ast,contextlib,io,json,re
import numpy as np
import sklearn
import tensorflow as tf

HERE=Path(__file__).resolve().parent
results={'tensorflow_version':tf.__version__,'sklearn_version':sklearn.__version__,'equations':{},'examples':[]}
# Step convention and XOR weights match the slides exactly.
step=lambda z:int(z>=0)
xor=[step(step(a+b-.5)-step(a+b-1.5)-.5) for a,b in [(0,0),(0,1),(1,0),(1,1)]]
assert xor==[0,1,1,0];results['equations']['xor']=xor
# Analytic chain rule agrees with an independent finite-difference calculation.
x,b,w,y=.8,.2,.7,1.
loss=lambda w:.5*(np.tanh(w*x+b)-y)**2
a=np.tanh(w*x+b);analytic=(a-y)*(1-a*a)*x
eps=1e-6;numeric=(loss(w+eps)-loss(w-eps))/(2*eps)
assert np.isclose(analytic,numeric,atol=1e-8)
results['equations']['chain_rule']={'analytic':float(analytic),'finite_difference':float(numeric)}
activation={'sigmoid_minus_one':float(1/(1+np.exp(1))),'tanh_minus_one':float(np.tanh(-1)),'relu_minus_one':0}
assert np.isclose(activation['sigmoid_minus_one'],.269,atol=.001)
assert np.isclose(activation['tanh_minus_one'],-.762,atol=.001)
results['equations']['activations']=activation
z=np.arange(10,dtype=float);softmax=np.exp(z-z.max());softmax/=softmax.sum()
assert np.isclose(softmax.sum(),1);results['equations']['softmax_sum']=float(softmax.sum())
source=next((HERE.parents[1]/'lectures').glob('lecture-29*.tex')).read_text()
frames=re.findall(r'\\begin\{frame\}\[fragile,t\]\{([^\n]+)\}(.*?)\\end\{frame\}',source,re.S)
env={'__name__':'__lecture_verification__'}
for title,body in frames:
 blocks=re.findall(r'\\begin\{verbatim\}(.*?)\\end\{verbatim\}',body,re.S)
 if not blocks:continue
 code='\n'.join(blocks);ast.parse(code)
 output=io.StringIO()
 with contextlib.redirect_stdout(output):exec(compile(code,'<lecture29>','exec'),env)
 entry={'title':title,'output':output.getvalue().strip()[-1200:]}
 if title.startswith('Prepare One Shared'):
  assert [len(env[k]) for k in ['X_train','X_val','X_test']]==[364,91,114]
  assert np.allclose(env['X_train'].mean(axis=0),0,atol=1e-10)
  assert env['scaler'].n_samples_seen_==364
  entry['partitions']=[364,91,114]
 if title=='Breast-Cancer Example: Define the Neural Network':
  assert env['model'].count_params()==4097
 if title=='Breast-Cancer Example: Fit for 30 Epochs':
  assert len(env['history'].history['loss'])==30
  with contextlib.redirect_stdout(output):metrics=env['model'].evaluate(env['X_test'],env['y_test'],verbose=0)
  assert all(np.isfinite(metrics)) and 0<=metrics[1]<=1
  entry['test_loss_accuracy']=[float(v) for v in metrics]
 if title=='Original Python Example: Learning a Sine Wave':
  assert len(env['model'].history.history['loss'])==100
  pred=env['model'](env['X'],training=False).numpy()
  assert pred.shape==env['y'].shape and np.all(np.isfinite(pred))
  entry['training_mse']=float(np.mean((pred-env['y'])**2))
 if title=='Original Python Example: A Dense MNIST Classifier':
  assert env['model'].count_params()==109386
  assert env['X_train'].shape==(60000,784) and env['X_test'].shape==(10000,784)
  assert env['X_train'].min()>=0 and env['X_train'].max()<=1
 if title=='MNIST: Train With Integer Labels and Test Once':
  assert len(env['X_train'])==50000 and len(env['X_val'])==10000
  assert len(env['model'].history.history['loss'])==3
  probabilities=env['model'](env['X_test'][:10],training=False).numpy()
  assert np.allclose(probabilities.sum(axis=1),1,atol=1e-6)
 results['examples'].append(entry)
 print('Passed:',title,flush=True)
(HERE/'example-checks.json').write_text(json.dumps(results,indent=2)+'\n')
print('All equation checks and full-budget demonstration runs passed.',flush=True)
