"""Run original saved-score checks without their two output writes. No fits/data access."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / 'research/time_to_high'

def verify():
    contract = json.loads((T / 'Results/staged/CONTRACT.json').read_text())
    expected = contract['identity']['hashes']
    names = {'reference':'screen_runner.py','optimized':'screen_throughput.py',
             'core':'staged_timing_core.py','runner':'staged_timing_runner.py'}
    sources = {key:(T/name).read_text(encoding='utf-8') for key,name in names.items()}
    pilot = json.loads((T/'Time_To_High_Screen_Pilot.ipynb').read_text())
    sources['range_reader'] = ''.join(pilot['cells'][1]['source']).split('CONFIG =',1)[0]
    for key, digest in expected.items():
        assert hashlib.sha256(sources[key].encode()).hexdigest() == digest, key
    path = T/'verify_staged_results.py'
    tree = ast.parse(path.read_text())
    removed = []
    class ReadOnly(ast.NodeTransformer):
        def visit_Expr(self, node):
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'write_text':
                removed.append(node.lineno)
                return None
            return node
    tree = ReadOnly().visit(tree)
    assert len(removed) == 2, 'Original verifier changed: review output behavior first'
    scope = {'__file__':str(path),'__name__':'__receipt_check__'}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),scope)
    return {'status':scope['receipt']['status'], 'actual_frozen_source_hashes':'PASS',
            'cheap_fold_scores_checked':scope['score_count'], 'selected_outer_folds':50,
            'source_write_calls_suppressed':removed, 'training_fits':0,
            'raw_data_replay':False, 'model_inference_replay':False,
            'limitations':scope['receipt']['limitations']}

if __name__ == '__main__':
    print(json.dumps(verify(),indent=2))
