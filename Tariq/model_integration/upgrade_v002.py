"""Download, replay and select v002; prune only the explicitly superseded local v001 files."""
import json
from pathlib import Path
import subprocess
import sys

from pipeline import ROOT, read, digest, inside, inspect_package, run


def verify_ready(config, result):
    import hashlib
    expected_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (result.get('status') != 'REPLAY_PASSED_DEVELOPMENT_ONLY'
            or result.get('prediction_source') != 'EXECUTED_FROZEN_MODELS'
            or result.get('config_sha256') != expected_hash
            or result.get('target_alignment') != 'MATCH'
            or result.get('evaluation_alignment', {}).get('status') != 'MATCH'):
        raise ValueError('Version 2 must pass real replay and alignment checks before replacement')
    for role in config['models']:
        if any(c['status'] != 'PASS' for c in inspect_package(config, role)):
            raise ValueError(role + ': files changed or are missing after replay')
        if {'check': 'prediction_replay', 'status': 'PASS'} not in result['checks'][role]:
            raise ValueError(role + ': replay evidence missing')


def old_file_plan(old, new):
    """Validate every deletion target and hash before promoting; no recursive deletion."""
    planned = []
    protected = [inside(ROOT, m[k]) for m in new['models'].values() for k in ('package_dir', 'artifact_dir')]
    for role, model in old['models'].items():
        if model['version'] == new['models'][role]['version']:
            continue
        if model['version'] not in ('sid_direction-15m-v001', 'sid_net-magnitude-15m-v001'):
            raise ValueError('This upgrade only removes the identified v001 submissions')
        package = inside(ROOT, model['package_dir'])
        artifact_dir = inside(ROOT, model['artifact_dir'])
        manifest = read(package / 'manifest.json')
        entries = [(inside(package, name), sha) for name, sha in model['package_sha256'].items()]
        entries += [(inside(artifact_dir, a['file']), a['sha256']) for a in manifest['artifacts']]
        for path, sha in entries:
            if any(path.is_relative_to(p) for p in protected):
                raise ValueError('Old file overlaps the new active package')
            if path.exists():
                if digest(path) != sha:
                    raise ValueError('Old file has unexpected changes; will not delete: ' + str(path))
                planned.append(path)
    return planned


def main():
    candidate_path = ROOT / 'config.v002.json'
    candidate = read(candidate_path)
    current_path = ROOT / 'config.json'
    old = read(current_path)
    model_python = ROOT / '.venv-model/Scripts/python.exe'
    if not model_python.exists():
        raise ValueError('Existing .venv-model is required')
    results = ROOT / 'results'
    results.mkdir(exist_ok=True)
    lock = results / 'runner.lock'
    with lock.open('x') as stream:
        stream.write('Version 2 upgrade active. Do not run a second upgrade or replay.\n')
    try:
        for role in candidate['models']:
            checks = inspect_package(candidate, role)
            if any(c['status'] == 'FAIL' and not c['check'].startswith('artifact:') for c in checks):
                raise ValueError(role + ': candidate documents failed verification')
        print('Step 1: retrieve v002 artifacts from MLflow. Password stays local.', flush=True)
        subprocess.run([sys.executable, str(ROOT / 'download_artifacts.py'), '--config', str(candidate_path)], check=True)
        print('Step 2: reuse the existing model environment; install missing pinned dependencies without cache.', flush=True)
        requirements = inside(ROOT, candidate['models']['direction']['package_dir']) / 'requirements.txt'
        other = inside(ROOT, candidate['models']['magnitude']['package_dir']) / 'requirements.txt'
        if requirements.read_bytes() != other.read_bytes():
            raise ValueError('Packages need separate dependency review')
        subprocess.run([str(model_python), '-m', 'pip', 'install', '--no-cache-dir', '-r', str(requirements)], check=True)
        print('Step 3: replay both fitted models and verify target/evaluation alignment.', flush=True)
        result = run(candidate, 'replay', str(model_python), results / 'current')
        verify_ready(candidate, result)
        planned = old_file_plan(old, candidate)
        # Keep a compact history, not a backup of retired models or packages.
        history_path = results / 'history.json'
        history = read(history_path) if history_path.exists() else []
        history.append({k: result.get(k) for k in ('created_at_utc', 'status', 'config_sha256', 'models',
                         'target_alignment', 'evaluation_alignment', 'predicted_signed_score_bps')})
        history_path.write_text(json.dumps(history[-100:], indent=2) + '\n')
        receipt_path = results / 'upgrade_receipt.json'
        receipt = {'status': 'VERIFIED_READY_TO_SELECT', 'old_versions': {r:m['version'] for r,m in old['models'].items()},
                   'new_versions': {r:m['version'] for r,m in candidate['models'].items()},
                   'planned_local_deletions': [str(p.relative_to(ROOT)) for p in planned],
                   'deleted_local_files': [], 'remote_records_changed': False}
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        temporary = ROOT / 'config.promoting.json'
        temporary.write_text(json.dumps(candidate, indent=2) + '\n')
        temporary.replace(current_path)
        for path in planned:
            path.unlink()
            receipt['deleted_local_files'].append(str(path.relative_to(ROOT)))
            receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        for folder in sorted({p.parent for p in planned}, key=lambda p: len(p.parts), reverse=True):
            if folder.exists() and not any(folder.iterdir()):
                folder.rmdir()
        receipt['status'] = 'V002_SELECTED_V001_LOCAL_FILES_REMOVED'
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        print(receipt['status'])
        print('Signed score (bps):', result['predicted_signed_score_bps'])
        print('Result:', results / 'current/result.json')
        print('Trading remains disabled. GitHub and MLflow history were not deleted.')
    finally:
        lock.unlink()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('Upgrade stopped:', error, file=sys.stderr)
        print('Inspect results/current/result.json and results/upgrade_receipt.json if present before retrying.', file=sys.stderr)
        sys.exit(1)
