"""Execute the lecture's displayed Python, checking figures and measured results.

Run with the dependencies in requirements.txt. Figures are generated separately
by generate-figures.py; this checks the actual code students see on the slides.
"""
from pathlib import Path
import contextlib
import io
import json
import re

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent.parent / 'lectures' / (
    'lecture-31-unsupervised-learning-clustering-dimensionality-reduction-learne.tex')
text = SOURCE.read_text()
expected = json.loads((ROOT / 'results.json').read_text())
blocks = re.findall(r'\\begin\{verbatim\}(.*?)\\end\{verbatim\}', text, re.S)
assert len(blocks) == 11
namespace = {}
records = []
for index, code in enumerate(blocks):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, f'{SOURCE.name}:example-{index+1}', 'exec'), namespace)
    record = {'example': index + 1, 'printed_output': output.getvalue().strip()}
    records.append(record)
    n = namespace
    lines = output.getvalue().strip().splitlines()
    if index == 0:
        result = expected['kmeans_blobs']
        assert [float(v) for v in lines] == [result[k] for k in ('inertia', 'silhouette', 'ari_posthoc')]
        assert np.bincount(n['cluster_id']).tolist() == result['cluster_sizes']
    elif index == 1:
        got = [[float(v) for v in line.split()] for line in lines]
        result = expected['k_selection']
        assert got == [list(row) for row in zip(result['ks'], result['inertia'], result['silhouette'])]
    elif index == 2:
        keys = ['kmeans_ari_posthoc', 'dbscan_ari_posthoc', 'kmeans_silhouette', 'dbscan_silhouette']
        assert [round(float(v), 3) for v in lines] == [expected['moons'][k] for k in keys]
        assert int((n['b'] == -1).sum()) == expected['moons']['dbscan_noise']
    elif index == 3:
        assert n['prob'].shape == (450, 3)
        assert np.allclose(n['prob'].sum(axis=1), 1)
    elif index == 4:
        assert np.allclose(n['pca'].explained_variance_ratio_, expected['pca_iris']['variance_ratio'], atol=0.00005)
        assert float(lines[1]) == expected['pca_iris']['first_two_total_pct']
    elif index == 5:
        got = {k: float(v) for k, v in (line.split() for line in lines)}
        assert got == expected['digits_reconstruction_mse']
        assert all(a > b for a, b in zip(got.values(), list(got.values())[1:]))
    elif index == 6:
        assert float(lines[0]) == expected['nmf']['relative_frobenius_reconstruction_error']
        assert n['W'].min() >= 0 and n['H'].min() >= 0
    elif index == 7:
        assert n['Z_pca'].shape == n['Z_tsne'].shape == (1150, 2)
        assert np.isfinite(n['Z_tsne']).all()
    elif index == 8:
        result = expected['digits_downstream_svc']
        assert [len(n[k]) for k in ['Xtr', 'Xval', 'Xtest']] == [result[k] for k in ['train_n', 'validation_n', 'test_n']]
    elif index == 9:
        result = expected['digits_downstream_svc']
        assert n['best_r'] is None and result['selected'] == '64 raw pixels'
        assert round(n['accuracy_score'](n['ytest'], n['final'].predict(n['Xtest'])), 4) == result['final_test_accuracy']
        assert [round(n['validation'][r], 4) for r in n['choices']] == list(result['validation_accuracy'].values())
    elif index == 10:
        assert float(lines[0]) == expected['one_class_svm']['normal_acceptance']
        assert int(lines[1]) == expected['one_class_svm']['anomalies_flagged']
    record['passed'] = True

figures = re.findall(r'\\lecturevisual(?:\[[^\]]+\])?\{([^}]+)\}', text)
assert len(set(figures)) == 18
assert all((ROOT / name).is_file() for name in figures)
assert '\\appendix' not in text
assert '\\placeholdernote' not in text
report = {'source': SOURCE.name, 'executed_examples': len(records),
          'unique_figures': len(set(figures)), 'all_checks_passed': True,
          'examples': records}
(ROOT / 'example-checks.json').write_text(json.dumps(report, indent=2) + '\n')
print(f'Passed: {len(records)} displayed Python examples and {len(set(figures))} figure references.')
