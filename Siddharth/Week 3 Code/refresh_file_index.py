"""Refresh the Week 3 path inventory; contents are not inspected."""
from pathlib import Path
import os
import csv
import json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
errors=[]
(ROOT/'FILE_INDEX_SUMMARY.json').touch(exist_ok=True)
with (ROOT/'FILE_INDEX.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.writer(f);writer.writerow(['relative_path','extension']);count=0
    for directory,dirs,files in os.walk(ROOT,followlinks=False,onerror=lambda e:errors.append(str(e))):
        dirs.sort()
        for name in sorted(files):
            path=Path(directory)/name
            writer.writerow([path.relative_to(ROOT).as_posix(),path.suffix.lower()]);count+=1
(ROOT/'FILE_INDEX_SUMMARY.json').write_text(json.dumps(dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),file_count=count,scope='Path inventory only; file contents not read; inaccessible paths logged.',traversal_errors=errors),indent=2),encoding='utf-8')
print(json.dumps(dict(files=count,traversal_errors=len(errors))))
