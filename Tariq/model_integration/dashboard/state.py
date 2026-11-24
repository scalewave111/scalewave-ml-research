"""Read-only dashboard data access. No model imports, network calls or file writes."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path, kind, warnings):
    try:
        data = path.read_bytes()
        if len(data) > 5_000_000:
            raise ValueError('record exceeds 5 MB')
        value = json.loads(data.decode('utf-8-sig'))
        if not isinstance(value, kind):
            raise ValueError('unexpected record structure')
        return value
    except (OSError, ValueError) as error:
        warnings.append(f'{path.name}: {error}')
        return kind()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def simulate(probability, magnitude, threshold):
    if not all(finite(x) for x in (probability, magnitude, threshold)):
        raise ValueError('Inputs must be finite numbers')
    if not 0 <= probability <= 1 or magnitude < 0 or threshold <= 0:
        raise ValueError('Invalid probability, magnitude or threshold')
    score = (2 * probability - 1) * magnitude
    interpretation = 'LONG candidate' if score > threshold else 'SHORT candidate' if score < -threshold else 'NO TRADE under this proposed filter'
    return {'score': score, 'interpretation': interpretation}


def monitor_status(record, now=None):
    if not record:
        return 'NOT_RECORDED'
    status = record.get('status', 'UNKNOWN')
    if status != 'RUNNING':
        return status
    try:
        heartbeat = datetime.fromisoformat(record['heartbeat_at_utc'].replace('Z','+00:00'))
        if heartbeat.tzinfo is None:
            return 'POSSIBLY_INTERRUPTED'
        age = ((now or datetime.now(timezone.utc)) - heartbeat).total_seconds()
        return 'POSSIBLY_INTERRUPTED' if age > 20 or age < -5 else 'RUNNING'
    except (KeyError, ValueError, TypeError):
        return 'POSSIBLY_INTERRUPTED'


def snapshot(root=ROOT):
    root = Path(root).resolve()
    warnings = []
    config = read_json(root / 'config.json', dict, warnings)
    result = read_json(root / 'results/current/result.json', dict, warnings)
    history = read_json(root / 'results/history.json', list, warnings)
    monitor_path = root / 'results/current/monitor.json'
    monitoring = read_json(monitor_path, dict, warnings) if monitor_path.exists() else {}
    observed_status = monitor_status(monitoring)
    expected_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    matches = bool(config) and result.get('config_sha256') == expected_hash
    if result and not matches:
        warnings.append('The saved result does not match the selected configuration. It is historical evidence only.')
    checks = []
    for role, items in result.get('checks', {}).items():
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    checks.append({'Model': role, 'Check': item.get('check', 'Unknown'),
                                   'Status': item.get('status', 'UNKNOWN'), 'Detail': item.get('detail', '')})
    models = config.get('models', {})
    outputs = result.get('outputs', {})
    direction = outputs.get('direction', {})
    magnitude = outputs.get('magnitude', {})
    p = direction.get('p_up')
    m = magnitude.get('net_magnitude_bps')
    score = result.get('predicted_signed_score_bps')
    identities = all(outputs.get(role, {}).get('model_version') == model.get('version') and
                     outputs.get(role, {}).get('symbol') == config.get('symbol') and
                     outputs.get(role, {}).get('horizon') == config.get('horizon')
                     for role, model in models.items()) and len(models) == 2
    valid_numbers = (finite(p) and 0 <= p <= 1 and finite(m) and m >= 0 and finite(score)
                     and math.isclose(score, (2*p-1)*m, abs_tol=1e-8, rel_tol=1e-8))
    timestamps_match = bool(direction.get('forecast_time_utc')) and direction.get('forecast_time_utc') == magnitude.get('forecast_time_utc')
    passed = result.get('status') == 'REPLAY_PASSED_DEVELOPMENT_ONLY'
    verified_outputs = bool(matches and passed and identities and valid_numbers and timestamps_match
                            and result.get('prediction_source') == 'EXECUTED_FROZEN_MODELS'
                            and all(any(c['Model'] == role and c['Check'] == 'prediction_replay' and c['Status'] == 'PASS'
                                        for c in checks) for role in ('direction', 'magnitude'))
                            and all(c['Status'] == 'PASS' for c in checks))
    if monitoring:
        verified_outputs = bool(verified_outputs and monitoring.get('attempt_id') == result.get('attempt_id')
                                and observed_status == 'COMPLETED'
                                and any(s.get('id') == 'save' and s.get('status') == 'PASSED' for s in monitoring.get('steps', [])))
    evaluations = {}
    for role, model in models.items():
        relative = model.get('package_dir', '')
        folder = (root / relative).resolve()
        if not relative or not folder.is_relative_to(root):
            warnings.append(role + ': invalid package path')
            continue
        evaluations[role] = read_json(folder / 'evaluation.json', dict, warnings)
    try:
        created = datetime.fromisoformat(result['created_at_utc'].replace('Z', '+00:00'))
        if created.tzinfo is None:
            raise ValueError('timestamp needs timezone')
        age_hours = max(0, (datetime.now(timezone.utc)-created).total_seconds()/3600)
    except (KeyError, ValueError, TypeError):
        age_hours = None
    return {'config': config, 'result': result, 'history': [h for h in history if isinstance(h, dict)][-100:],
            'matches': matches, 'checks': checks, 'verified_outputs': verified_outputs,
            'p_up': p if verified_outputs else None, 'magnitude': m if verified_outputs else None,
            'score': score if verified_outputs else None, 'evaluations': evaluations,
            'warnings': warnings, 'age_hours': age_hours, 'monitoring': monitoring, 'monitor_status': observed_status}
