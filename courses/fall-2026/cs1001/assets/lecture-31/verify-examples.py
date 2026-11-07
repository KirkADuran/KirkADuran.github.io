"""Execute the slide code and check the supplied array states and trace tables."""
from pathlib import Path
import ast
import contextlib
import io
import json
import re
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent.parent / 'lectures/lecture-31-sorting-algorithms.tex'
text = SOURCE.read_text()
blocks = re.findall(r'\\begin\{verbatim\}(.*?)\\end\{verbatim\}', text, re.S)
assert len(blocks) == 15
namespace = {}
for i, code in enumerate(blocks):
    compile(code, f'{SOURCE.name}:example-{i}', 'exec')
for index in [0, 3, 8, 10]:
    exec(compile(blocks[index], f'{SOURCE.name}:example-{index}', 'exec'), namespace)

checks = []
cases = [[], [7], [1, 2, 3], [3, 2, 1], [2, 2, 1, 2],
         [-4, 0, 3, -9, 0], [5, 2, 4, 1, 3]]
for name in ['selection_sort', 'insertion_sort']:
    f = namespace[name]
    for case in cases:
        values = list(case)
        alias = values
        assert f(values) is None
        assert values is alias and values == sorted(case), (name, case, values)
    checks.append(f'{name}: correct in-place result and None return for {len(cases)} edge/example inputs')

# Capture the actual classroom function, including values at the next outer-loop visit.
def trace_function(name, variable):
    f = namespace[name]
    outer = []
    inner = []
    def trace(frame, event, arg):
        if frame.f_code is f.__code__ and event == 'line':
            n = frame.f_locals
            if frame.f_lineno == 3 and variable in n:
                outer.append({variable: n[variable], 'key': n.get('key'),
                              'min_index': n.get('min_index'), 'values': list(n['values'])})
            if name == 'selection_sort' and frame.f_lineno == 5 and n.get('start') == 0 and 'scan' in n:
                inner.append((n['scan'], n['min_index']))
            if name == 'insertion_sort' and frame.f_lineno == 6 and n.get('key') == 1:
                inner.append(list(n['values']))
        return trace
    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        f([5, 2, 4, 1, 3])
    finally:
        sys.settrace(previous)
    return outer, inner

# Verbatim starts with a newline, so function and loop are lines 2 and 3.
selection, selection_inner = trace_function('selection_sort', 'start')
insertion, insertion_inner = trace_function('insertion_sort', 'position')
def frame_body(title):
    return text.split('{' + title + '}', 1)[1].split('\\end{frame}', 1)[0]
sel_rows = re.findall(r'^(\d+) & (\d+) & .*?\\texttt\{(\[[^}]+\])\}',
                      frame_body('Trace the Code: Selection Sort, Outer Loop'), re.M)
assert [(d['start'], d['min_index'], d['values']) for d in selection] == [
    (int(a), int(b), ast.literal_eval(c)) for a, b, c in sel_rows]
ins_rows = re.findall(r'^(\d+) & (\d+) & .*?\\texttt\{(\[[^}]+\])\}',
                      frame_body('Trace the Code: Insertion Sort, Outer Loop'), re.M)
assert [(d['position'], d['key'], d['values']) for d in insertion] == [
    (int(a), int(b), ast.literal_eval(c)) for a, b, c in ins_rows]
assert selection_inner == [(1, 1), (2, 1), (3, 3), (4, 3)], selection_inner
manifest = json.loads((ROOT / 'supplied-images.json').read_text())
arrays = {p['path']: p for p in manifest['images']}
assert insertion_inner == [arrays[f'array-{i:02}.png']['values'] for i in [11, 12, 13, 14]], insertion_inner
assert insertion[2]['values'] == arrays['array-10.png']['values']
checks.append('Both outer-loop tables, first selection scan, and insertion-shift diagrams match execution')

namespace['values'] = [5, 2, 4, 1, 3]
exec(blocks[1], namespace)
assert namespace['min_index'] == 3
exec(blocks[2], namespace)
assert namespace['values'] == [1, 2, 4, 5, 3]
namespace['values'] = [5, 2, 4, 1, 3]
exec(blocks[4], namespace)
assert namespace['values'] == [2, 5, 4, 1, 3]
for i in [5, 6, 7, 9, 11, 12, 13, 14]:
    with contextlib.redirect_stdout(io.StringIO()):
        exec(blocks[i], namespace)
    if i == 5:
        assert namespace['values'] == [1, 2, 3, 4, 5]
    elif i == 6:
        assert namespace['ordered'] == [1, 2, 3, 4, 5]
        assert namespace['values'] == [5, 2, 4, 1, 3]
    elif i == 7:
        assert namespace['ordered'] == namespace['values'] == [5, 4, 3, 2, 1]
    elif i == 9:
        assert namespace['ordered'] == ['Bo', 'Ada', 'Christopher']
    elif i == 12:
        original = list(namespace['students'])
    elif i == 13:
        assert [s.name for s in namespace['by_gpa']] == ['Maya', 'Nora', 'Luis']
        assert all(a is b for a, b in zip(namespace['students'], original))
        assert namespace['by_gpa'][0] is original[0]
    elif i == 14:
        assert [s.name for s in namespace['by_name']] == ['Luis', 'Maya', 'Nora']
        assert [s.name for s in namespace['students']] == ['Luis', 'Nora', 'Maya']
        assert [s.name for s in namespace['by_id']] == ['Maya', 'Luis', 'Nora']
assert namespace['get_gpa'](original[1]) == 3.8
Student = namespace['Student']
ties = [Student(1, 'First', 3.8), Student(2, 'Second', 3.8)]
assert sorted(ties, key=lambda s: s.gpa, reverse=True) == ties
checks.append('Built-in sorting, named/lambda keys, object identity, ascending/descending orders, and stability pass')
figures = set(re.findall(r'\\arraydiagram(?:\[[^\]]+\])?\{(\d+)\}', text))
assert len(figures) == 14
assert all((ROOT / f'array-{n}.png').exists() for n in figures)
report = {'source': SOURCE.name, 'code_blocks_compiled': len(blocks), 'all_checks_passed': True,
          'checks': checks, 'selection_trace': selection, 'insertion_trace': insertion,
          'supplied_array_diagrams': len(figures)}
(ROOT / 'example-checks.json').write_text(json.dumps(report, indent=2) + '\n')
print('Passed: 15 code blocks, sorting boundary cases, trace tables, object sorting, and 14 supplied diagrams.')
