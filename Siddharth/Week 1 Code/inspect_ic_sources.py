from pathlib import Path
import json, importlib.util
ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'Report' / 'ic_rf_sources'
out.mkdir(exist_ok=True)
print({m: bool(importlib.util.find_spec(m)) for m in ['pypdf','pdfplumber','fitz','duckdb','sklearn','pandas']})
from pypdf import PdfReader
paths = list((ROOT/'Report').glob('D[68]*.pdf')) + list((Path.home()/'Downloads').glob('Top 20*magnitude*.pdf')) + list((Path.home()/'Downloads').glob('Top 20*MAGNITUDE*.pdf'))
for p in dict.fromkeys(paths):
    r=PdfReader(p)
    dest=out/(p.stem+'.txt')
    dest.write_text('\n\n'.join(f'PAGE {i+1}\n'+(page.extract_text() or '') for i,page in enumerate(r.pages)),encoding='utf-8')
    print(json.dumps({'source':str(p),'pages':len(r.pages),'text':str(dest)},ensure_ascii=True))
