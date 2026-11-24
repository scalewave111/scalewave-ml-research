"""Local development replay of a pinned direction/magnitude pair. No execution APIs."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from monitoring import Monitor, atomic_json, error_text

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path escapes configured directory')
    return path


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Expected a finite number (not a boolean)')
    return value


def timestamp(value):
    when = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if when.tzinfo is None or when.utcoffset() is None:
        raise ValueError('Timezone-aware timestamp required')
    return when.astimezone(timezone.utc)


def validate_identity(row, config, model):
    for key in ('symbol', 'horizon'):
        if row[key] != config[key]:
            raise ValueError('Unexpected ' + key)
    when = timestamp(row['forecast_time_utc'])
    grid = config['forecast_grid_minutes']
    if when.second or when.microsecond or (when.hour * 60 + when.minute) % grid:
        raise ValueError('Forecast is outside the configured UTC issuance grid')
    if 'model_version' in row and row['model_version'] != model['version']:
        raise ValueError('Output model version does not match selected package')
    return when


def input_columns(path):
    manifest = read(path.parent / 'manifest.json')
    if 'input_features' in manifest:
        return tuple(manifest['input_features'])
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'INPUT_COLUMNS' for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise ValueError('Predictor must declare static INPUT_COLUMNS; review adapter for new package')


def validate_input(row, columns, config, model):
    when = validate_identity(row, config, model)
    if not isinstance(row['features'], dict) or set(row['features']) != set(columns):
        raise ValueError('Feature names do not match this predictor')
    for value in row['features'].values():
        if value is not None:
            number(value)
    if model.get('requires_scale_bps') and number(row.get('scale_bps')) <= 0:
        raise ValueError('scale_bps must be positive')
    return when


def validate_output(row, config, model):
    if row.get('model_version') != model['version']:
        raise ValueError('Missing or incorrect output model version')
    when = validate_identity(row, config, model)
    value = number(row[model['output_field']])
    if model['output_field'] == 'p_up' and not 0 <= value <= 1:
        raise ValueError('Probability outside [0, 1]')
    if model['output_field'] == 'net_magnitude_bps' and value < 0:
        raise ValueError('Negative absolute magnitude')
    return when


def combine(direction, magnitude, config):
    d = validate_output(direction, config, config['models']['direction'])
    m = validate_output(magnitude, config, config['models']['magnitude'])
    if d != m:
        raise ValueError('Forecast timestamps differ')
    return number((2 * direction['p_up'] - 1) * magnitude['net_magnitude_bps'])


def inspect_package(config, role):
    model = config['models'][role]
    package = inside(ROOT, model['package_dir'])
    artifacts = inside(ROOT, model['artifact_dir'])
    checks = []
    def check(name, action):
        try:
            action()
            checks.append({'check': name, 'status': 'PASS'})
        except Exception as error:
            checks.append({'check': name, 'status': 'FAIL', 'detail': str(error)})
    def equal(actual, expected, label):
        if actual != expected:
            raise ValueError(label + ' differs from pinned value')
    for name, sha in model['package_sha256'].items():
        check('package:' + name, lambda n=name, s=sha: equal(digest(inside(package, n)), s, n))
    manifest = read(package / 'manifest.json')
    check('manifest_identity', lambda: equal(
        [manifest[k] for k in ('model_version', 'symbol', 'horizon', 'target', 'training_cutoff_utc')],
        [model['version'], config['symbol'], config['horizon'], model['target'], model['training_cutoff_utc']], 'Manifest'))
    if 'output_units' in manifest:
        check('output_units', lambda: equal(manifest['output_units'], model['units'], 'Units'))
    else:
        check('target_definition', lambda: equal(manifest['target_definition'], model['target_definition'], 'Target definition'))
        check('output_key', lambda: equal(manifest['output_key'], model['output_field'], 'Output key'))
    for artifact in manifest['artifacts']:
        check('artifact:' + artifact['file'], lambda a=artifact: equal(
            digest(inside(artifacts, a['file'])), a['sha256'], a['file']))
    example = read(package / 'example_input.json')
    check('example_input', lambda: validate_input(example, input_columns(package / 'predict.py'), config, model))
    check('expected_output', lambda: validate_output(read(package / 'expected_output.json'), config, model))
    check('example_timestamp_match', lambda: equal(timestamp(example['forecast_time_utc']),
          timestamp(read(package / 'expected_output.json')['forecast_time_utc']), 'Example timestamp'))
    return checks


def evaluation_alignment(config):
    """Compare supplied summaries; does not assert row-level identity or rerun evaluation."""
    reports = [read(inside(ROOT, m['package_dir']) / 'evaluation.json') for m in config['models'].values()]
    keys = ('evaluation_start_utc', 'evaluation_end_utc', 'row_count')
    if any(any(key not in r for key in keys) for r in reports):
        return {'status': 'NOT_RECORDED'}
    summaries = [{key: r[key] for key in keys} for r in reports]
    if any(any(not all(k in q for k in ('issue', 'test_last', 'test_rows'))
                   for q in r.get('quarterly_results', [])) for r in reports):
        return {'status': 'NOT_RECORDED', 'summaries': summaries}
    quarters = [[(timestamp(q['issue']).isoformat(), q['test_last'], q['test_rows'])
                 for q in r.get('quarterly_results', [])] for r in reports]
    matched = summaries[0] == summaries[1] and quarters[0] == quarters[1]
    return {'status': 'MATCH' if matched else 'MISMATCH', 'summaries': summaries,
            'evidence': 'Submitted dates and counts only; row-level identity not independently verified'}


def environment_check(python, requirements, version):
    probe = "import json,platform,importlib.metadata as m; print(json.dumps({'python':platform.python_version(),'packages':{x:m.version(x) for x in " + repr(list(requirements)) + "}}))"
    result = subprocess.run([python, '-c', probe], check=True, capture_output=True, text=True, timeout=30)
    actual = json.loads(result.stdout)
    if actual != {'python': version, 'packages': requirements}:
        raise ValueError('Model environment does not match pinned Python/dependency versions: ' + json.dumps(actual))


def run(config, mode, python, output_dir):
    start = time.monotonic()
    if Path(python).is_file():
        python = str(Path(python).resolve())
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(ROOT):
        raise ValueError('Results must stay inside the integration workspace')
    output_dir.mkdir(parents=True, exist_ok=True)
    # Clear only known generated predictions; a failed attempt must not expose old outputs.
    for name in ('direction_output.json', 'magnitude_output.json'):
        target = inside(output_dir, name)
        if target.exists():
            target.unlink()
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'mode': mode,
              'status': 'STARTED', 'trading_enabled': False, 'scope': 'DEVELOPMENT_ONLY',
              'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
              'formula_version': config['formula_version'], 'formula': '(2 * p_up - 1) * net_magnitude_bps',
              'symbol': config['symbol'], 'horizon': config['horizon'], 'models': {}, 'checks': {},
              'target_windows': {r: m['target_window'] for r, m in config['models'].items()},
              'blockers': ['No trading approval or execution component', 'Live feature pipeline not implemented',
                           'Original training commits unavailable', 'Current-market performance unverified']}
    windows = list(result['target_windows'].values())
    result['target_alignment'] = 'MATCH' if windows[0] == windows[1] else 'MISMATCH'
    if result['target_alignment'] == 'MISMATCH':
        result['blockers'].append('Direction and magnitude refer to different target windows')
    monitor = Monitor(output_dir, config, mode)
    result['attempt_id'] = monitor.record['attempt_id']
    final_status = 'FAILED'
    try:
        atomic_json(output_dir / 'result.json', result)
        with monitor.step('alignment'):
            result['evaluation_alignment'] = evaluation_alignment(config)
            if config.get('require_aligned_pair') and (result['target_alignment'] != 'MATCH' or result['evaluation_alignment']['status'] != 'MATCH'):
                raise ValueError('Selected pair fails required target/evaluation alignment')
        with monitor.step('files'):
            for role, model in config['models'].items():
                result['models'][role] = {k: model[k] for k in ('version', 'run_id', 'handoff_commit', 'training_cutoff_utc')}
                result['checks'][role] = inspect_package(config, role)
            failures = [role + ': ' + c['check'] + ': ' + c.get('detail', '')
                        for role, checks in result['checks'].items() for c in checks
                        if c['status'] == 'FAIL' and c['check'] not in ('example_input','expected_output','example_timestamp_match')
                        and (mode != 'reported-example' or not c['check'].startswith('artifact:'))]
            if failures:
                raise ValueError('; '.join(failures))
        with monitor.step('inputs'):
            failures = [role + ': ' + c['check'] + ': ' + c.get('detail', '')
                        for role, checks in result['checks'].items() for c in checks
                        if c['status'] == 'FAIL' and c['check'] in ('example_input','expected_output','example_timestamp_match')]
            if failures:
                raise ValueError('; '.join(failures))
            origins = []
            for role, model in config['models'].items():
                package = inside(ROOT, model['package_dir'])
                origins.append(validate_input(read(package / 'example_input.json'), input_columns(package / 'predict.py'), config, model))
            if origins[0] != origins[1]:
                raise ValueError('Example forecast timestamps differ between models')
        outputs = {}
        if mode == 'check':
            for key in ('direction','magnitude','combine'):
                monitor.skip(key, 'File-check mode does not execute predictions')
            result['status'] = 'FILES_READY_REPLAY_NOT_RUN'
        else:
            for role, model in config['models'].items():
                package = inside(ROOT, model['package_dir'])
                expected = read(package / 'expected_output.json')
                if mode == 'reported-example':
                    monitor.skip(role, 'Formula demonstration reads expected outputs; no model execution')
                    outputs[role] = expected
                    continue
                with monitor.step(role):
                    requirements = dict(line.strip().split('==') for line in (package / 'requirements.txt').read_text().splitlines()
                                        if line.strip() and not line.startswith('#'))
                    environment_check(python, requirements, read(package / 'manifest.json')['python_version'])
                    destination = output_dir / (role + '_output.json')
                    process = subprocess.run([python, str(package / 'predict.py'), '--input', str(package / 'example_input.json'),
                                              '--output', str(destination), '--artifact-dir', str(inside(ROOT, model['artifact_dir']))],
                                             cwd=package, capture_output=True, text=True, timeout=120)
                    if process.returncode:
                        raise ValueError(role + ': predictor failed (exit ' + str(process.returncode) + '); inspect package/environment')
                    actual = read(destination)
                    when = validate_output(actual, config, model)
                    if when != timestamp(expected['forecast_time_utc']):
                        raise ValueError(role + ': replay timestamp differs')
                    tolerance = model.get('replay_tolerance') or read(package / 'evaluation.json')['replay_tolerance']
                    if not math.isclose(actual[model['output_field']], expected[model['output_field']],
                                        rel_tol=tolerance['relative'], abs_tol=tolerance['absolute']):
                        raise ValueError(role + ': prediction replay mismatch')
                    result['checks'][role].append({'check': 'prediction_replay', 'status': 'PASS'})
                    outputs[role] = actual
            with monitor.step('combine'):
                result['predicted_signed_score_bps'] = combine(outputs['direction'], outputs['magnitude'], config)
                result['forecast_time_utc'] = outputs['direction']['forecast_time_utc']
                result['outputs'] = outputs
                result['prediction_source'] = 'SUBMITTED_EXPECTED_OUTPUTS_ONLY' if mode == 'reported-example' else 'EXECUTED_FROZEN_MODELS'
                result['status'] = 'FORMULA_DEMO_NOT_MODEL_REPLAY' if mode == 'reported-example' else 'REPLAY_PASSED_DEVELOPMENT_ONLY'
        final_status = 'COMPLETED'
    except (Exception, KeyboardInterrupt) as error:
        result['status'] = 'INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED'
        final_status = result['status']
        result['error'] = error_text(error)
        for key in ('outputs','predicted_signed_score_bps','prediction_source'):
            result.pop(key, None)
    finally:
        result['elapsed_seconds'] = round(time.monotonic() - start, 4)
        try:
            with monitor.step('save'):
                atomic_json(output_dir / 'config.json', config)
                atomic_json(output_dir / 'result.json', result)
        except (Exception, KeyboardInterrupt):
            final_status = 'FAILED'
            raise
        finally:
            monitor.finish(final_status)

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['check', 'reported-example', 'replay'], default='check')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--model-python', default=sys.executable)
    args = parser.parse_args()
    output = ROOT / 'results' / 'current'
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / 'runner.lock'
    try:
        with lock.open('x', encoding='utf-8') as stream:
            stream.write('Runner active. Remove only after confirming no runner remains.\n')
    except FileExistsError:
        parser.error('Another run may be active; inspect results/runner.lock before retrying')
    try:
        result = run(read(args.config), args.mode, args.model_python, output)
        history_path = output.parent / 'history.json'
        history = read(history_path) if history_path.exists() else []
        history.append({k: result.get(k) for k in ('created_at_utc', 'mode', 'status', 'config_sha256',
                        'models', 'target_alignment', 'predicted_signed_score_bps', 'elapsed_seconds')})
        history_path.write_text(json.dumps(history[-100:], indent=2) + '\n', encoding='utf-8')
    finally:
        lock.unlink()
    print(result['status'])
    print('Target alignment:', result['target_alignment'], '| Trading enabled: False')
    if 'predicted_signed_score_bps' in result:
        print('Signed score (bps):', result['predicted_signed_score_bps'])
    print('Result:', output / 'result.json')
    return 1 if result['status'] in ('FAILED', 'BLOCKED', 'INTERRUPTED') else 0


if __name__ == '__main__':
    sys.exit(main())
