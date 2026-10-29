"""Execute the lecture's actual Python blocks and record validation/test results.

Requires scikit-learn >= 1.4, numpy, matplotlib. Set SCIKIT_LEARN_DATA to reuse a
California Housing cache. The public dataset is downloaded if not cached.
Results here are measured; the explicitly illustrative slide plots are separate.
"""
from pathlib import Path
import ast, contextlib, io, json, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn

HERE=Path(__file__).resolve().parent
source=next((HERE.parents[1]/'lectures').glob('lecture-28*.tex')).read_text()
frames=re.findall(r'\\begin\{frame\}\[fragile,t\]\{([^\n]+)\}(.*?)\\end\{frame\}',source,re.S)
assert len(frames)==4
results={'sklearn_version':sklearn.__version__,'examples':[]}
env={'__name__':'__lecture_verification__'}
plt.show=lambda:plt.close('all')
for title,body in frames:
 blocks=re.findall(r'\\begin\{verbatim\}(.*?)\\end\{verbatim\}',body,re.S)
 code='\n'.join(blocks);ast.parse(code)
 stdout=io.StringIO()
 with contextlib.redirect_stdout(stdout):exec(compile(code,'<lecture28>','exec'),env)
 entry={'title':title,'output':stdout.getvalue().strip()}
 if 'scores' in env and title.startswith('Wine:'):
  assert len(env['scores'])==2 and all(0<=x<=1 for x in env['scores'])
  assert env['best'] is env['models'][env['scores'].index(max(env['scores']))]
  assert len(env['X_tr'])+len(env['X_v'])+len(env['X_test'])==178
  entry['partition_sizes']=[len(env[n]) for n in ['X_tr','X_v','X_test']]
 if 'errors' in env and title.startswith('Housing:'):
  assert len(env['errors'])==2 and all(x>=0 for x in env['errors'])
  assert env['best'] is env['models'][env['errors'].index(min(env['errors']))]
  assert len(env['X_tr'])+len(env['X_v'])+len(env['X_test'])==20640
  entry['partition_sizes']=[len(env[n]) for n in ['X_tr','X_v','X_test']]
 results['examples'].append(entry)
 print(title,':',entry['output'] or 'tree plot executed')
(HERE/'example-checks.json').write_text(json.dumps(results,indent=2)+'\n')
print('All four executable slide examples passed on the actual datasets.')
