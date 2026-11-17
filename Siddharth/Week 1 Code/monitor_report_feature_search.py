"""Read-only progress for the feature search; never reads test scores."""
import json,sys,time
from pathlib import Path
from datetime import datetime, timezone

root=Path(__file__).resolve().parent/'experiments/report38_subset_search_v2'
previous=None
while True:
    rows=[]
    for path in sorted(root.glob('*/*/STATUS.json')):
        try:status=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError):continue
        rows.append(f"{path.parent.relative_to(root)} {status['stage']} {status.get('completed','')}/{status.get('total',status.get('limit',''))}")
    rows.append(f"Audited runs: {len(list(root.glob('*/*/DONE.json')))} / 8")
    state='\n'.join(rows)
    if state!=previous:
        print(datetime.now(timezone.utc).isoformat()+'\n'+state,flush=True);previous=state
    if '--watch' not in sys.argv or (root/'COMPLETE.json').exists():break
    time.sleep(30)
