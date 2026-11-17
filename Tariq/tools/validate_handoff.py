"""Read-only package checks: hashes, authored links, notebook safety, source portability."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def validate():
    sm=json.loads((ROOT/'artifacts/source_map.json').read_text(encoding='utf-8'))
    copied={v['path']:v for v in sm['files']}
    errors=[]
    def check(ok,message):
        if not ok:errors.append(message)
    for rel,v in copied.items():
        p=ROOT/rel
        check(p.is_file() and sha(p)==v['packaged_sha256'],'Packaged source hash: '+rel)
    link_count=0; notebook_count=0; scanned=0
    secret_patterns=[r'gh[pousr]_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{40,}',
        r'AIza[0-9A-Za-z_-]{30,}',r'AKIA[0-9A-Z]{16}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        r'(?i)(?:api_key|access_token|refresh_token|client_secret)\s*[=:]\s*[\x22\x27][A-Za-z0-9_./+-]{20,}']
    for p in ROOT.rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ('.git','.validation','__pycache__')) or any(x.startswith('.venv') for x in p.parts):continue
        rel=p.relative_to(ROOT).as_posix()
        check(p.stat().st_size<100*1024**2,'GitHub individual file limit: '+rel)
        check(p.suffix.lower() not in ('.zip','.pem','.key'), 'Excluded file type: '+rel)
        if p.suffix in ('.json','.csv','.md','.txt','.html','.py','.ipynb'):
            s=p.read_text(encoding='utf-8-sig');scanned+=1
            # The scanner contains pattern literals, not credentials; no matches printed.
            for pattern in secret_patterns:
                check(re.search(pattern,s) is None,'Possible credential pattern in '+rel)
        if p.suffix=='.py':
            try:ast.parse(p.read_text(encoding='utf-8-sig'))
            except SyntaxError:errors.append('Python syntax: '+rel)
        if p.suffix=='.ipynb':
            notebook_count+=1;j=json.loads(p.read_text(encoding='utf-8-sig'))
            check(all(not c.get('outputs') for c in j['cells']),'Embedded notebook outputs: '+rel)
            check(not j.get('metadata',{}).get('widgets'),'Embedded notebook widget state: '+rel)
        if p.suffix=='.md' and rel not in copied:
            s=re.sub(r'```.*?```','',p.read_text(encoding='utf-8'),flags=re.S)
            for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',s):
                target=link.strip('<>').split('#')[0]
                if not target or re.match(r'[a-zA-Z]+://',target):continue
                link_count+=1
                check((p.parent/unquote(target)).exists(),f'Broken authored link: {rel} -> {target}')
    # Evaluate only the ROOT assignment, without running either builder/test.
    for name in ['build_alternative_notebook.py','test_alternative_notebook.py']:
        p=ROOT/'research/feature_exploration/Notebooks'/name
        node=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ROOT' for t in n.targets))
        resolved=eval(compile(ast.Expression(node.value),str(p),'eval'),{'Path':Path,'__file__':str(p)})
        check(resolved==ROOT/'research/feature_exploration','Portable helper root: '+name)
    manifest_path=ROOT/'artifacts/manifest.json'
    manifest_count=0
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        for v in manifest['files']:
            manifest_count+=1;p=ROOT/v['path']
            check(p.is_file() and sha(p)==v['sha256'],'Manifest hash: '+v['path'])
    result={'status':'PASS' if not errors else 'FAIL','copied_files_checked':len(copied),
            'authored_links_checked':link_count,'notebooks_checked':notebook_count,
            'text_files_scanned':scanned,'manifest_files_checked':manifest_count,
            'credential_scan':'Pattern-based; not a guarantee that every sensitive value can be detected',
            'historical_links':'Original reports/manifests retain historical references; not all are portable or available',
            'errors':errors}
    print(json.dumps(result,indent=2))
    return not errors

if __name__=='__main__':sys.exit(0 if validate() else 1)
