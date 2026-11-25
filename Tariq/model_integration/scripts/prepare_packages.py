"""On the execution host, restore exact package documents from a pinned Git commit."""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
from run import ROOT, read, inside, digest


def prepare(config, source_repo):
    ref = config['package_source_ref']
    if not re.fullmatch(r'[0-9a-f]{40}', ref):
        raise ValueError('A full pinned commit SHA is required')
    repo = Path(source_repo).resolve()
    if not repo.is_dir():
        raise ValueError('Source checkout does not exist')
    pending = []
    for role, model in config['models'].items():
        for name, expected in model['package_sha256'].items():
            target = inside(inside(ROOT, model['package_dir']), name)
            if target.exists():
                if digest(target) != expected:
                    raise ValueError('Existing package differs: ' + role + '/' + name)
                continue
            source = model['source_dir'].rstrip('/') + '/' + name
            result = subprocess.run(['git', '-C', str(repo), 'show', ref + ':' + source],
                                    capture_output=True, timeout=30)
            if result.returncode:
                raise ValueError('Pinned source unavailable: ' + source +
                                 '. Fetch the configured commit into the source checkout first.')
            if hashlib.sha256(result.stdout).hexdigest() != expected:
                raise ValueError('Source package hash mismatch: ' + role + '/' + name)
            pending.append((target, result.stdout))
    # Validate the complete source set before writing any new package documents.
    for target, content in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(content)
    return len(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-repo', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/models.json')
    args = parser.parse_args()
    count = prepare(read(args.config), args.source_repo)
    print(f'Prepared {count} pinned package files. Existing matching files were reused.')
    print('Next: scripts/download_models.py, then install model requirements and verify replay.')


if __name__ == '__main__':
    main()
