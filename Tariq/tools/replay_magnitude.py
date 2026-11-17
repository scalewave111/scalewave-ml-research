"""Replay unchanged magnitude source using approved external inputs; never fit models."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
P = ROOT/'research/magnitude_48h/playground'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True,help='Approved folder containing the four prepared X/y Parquet files')
    args=parser.parse_args()
    data=args.data_root.resolve()
    config=json.loads((P/'submissions/magnitude_current/run_001/config.json').read_text())
    hashes=json.loads((P/'submissions/magnitude_current/run_001/input_hashes.json').read_text())
    mapping={}
    for part in config['data_parts']:
        for rel in part.values():
            target=data/Path(rel).name
            expected=hashes.get(rel.replace('/','\\'),hashes.get(rel))
            assert expected and digest(target)==expected, f'Input hash mismatch: {target.name}'
            mapping[(P/rel).resolve()]=target
    # Verify current packaged source/model/card/prediction bytes before loading joblib.
    manifest=json.loads((ROOT/'artifacts/source_map.json').read_text())
    for item in manifest['files']:
        if '/magnitude_current/' in item['path'] or '/magnitude_walkforward_v2/run_001/fold_' in item['path']:
            assert digest(ROOT/item['path'])==item['packaged_sha256'],item['path']
    import pandas as pd
    original=pd.read_parquet
    def read_external(path,*a,**kw):
        return original(mapping.get(Path(path).resolve(),path),*a,**kw)
    pd.read_parquet=read_external
    try:
        runpy.run_path(str(P/'submissions/magnitude_current/replay.py'),run_name='__main__')
    finally:
        pd.read_parquet=original

if __name__=='__main__':main()
