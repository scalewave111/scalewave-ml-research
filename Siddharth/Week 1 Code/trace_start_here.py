from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
import re,json,hashlib,collections
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'Dataset/TEAM_48H_PLAYGROUND_V1/TEAM_48H_PLAYGROUND_V1'
OUT=ROOT/'Report/start_here_audit'
OUT.mkdir(exist_ok=True)
class H(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.parts=[];self.skip=0
 def handle_starttag(self,t,a):
  if t in ('style','script'):self.skip+=1
  if t=='a':self.links += [v for k,v in a if k=='href']
 def handle_endtag(self,t):
  if t in ('style','script'):self.skip=max(0,self.skip-1)
  if t in ('p','div','li','tr','h1','h2','h3','section'):self.parts.append('\n')
 def handle_data(self,d):
  if not self.skip:self.parts.append(d)
queue=collections.deque([BASE/'START_HERE.html'])
seen=set(); records=[]; edges=[]
pattern=r'(?:[A-Za-z0-9_.-]+[/\\])*[A-Za-z0-9_.-]+\.(?:json|csv|md|html|py|parquet|txt)'
while queue:
 p=queue.popleft().resolve()
 if p in seen:continue
 seen.add(p)
 if not p.is_file():continue
 suffix=p.suffix.lower()
 if suffix not in ['.html','.md','.json','.csv','.py','.txt']:
  records.append(dict(path=str(p),status='present_not_text_parsed'));continue
 raw=p.read_text(encoding='utf-8-sig',errors='replace')
 record=dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),status='text_read')
 if suffix=='.html':
  h=H();h.feed(raw);links=h.links;text=''.join(h.parts)
 elif suffix=='.md':
  links=re.findall(r'\[[^\]]*\]\(([^)]+)\)',raw);text=raw
 else:links=[];text=raw
 if suffix=='.json':
  try: obj=json.loads(raw);record['json_type']=type(obj).__name__
  except Exception as exc:record['parse_error']=str(exc)
 if suffix=='.csv':
  record['header']=raw.splitlines()[0] if raw else ''
  record['data_lines']=max(0,len(raw.splitlines())-1)
 # Explicit links plus concrete filenames mentioned by the documents/code.
 refs=[(v,'link') for v in links]
 if suffix in ['.html','.md','.py','.json']:
  refs += [(v,'path_reference') for v in re.findall(pattern,text)]
 record['relevant_lines']=[dict(line=i+1,text=s) for i,s in enumerate(text.splitlines()) if re.search(r'train|validat|develop|holdout|purge|split|cutoff|test|fold',s,re.I)]
 records.append(record)
 for ref,kind in dict.fromkeys(refs):
  ref=unquote(ref.strip().strip('<>').split('#')[0]).replace('\\','/')
  if re.match(r'^/[A-Za-z]:/',ref):ref=ref[1:]
  if not ref:continue
  if ref.startswith(('http:','https:','mailto:','javascript:')):
   edges.append(dict(source=str(p),target=ref,kind=kind,status='external_url'));continue
  if ref.startswith('file:'):ref=unquote(urlparse(ref).path).lstrip('/')
  candidates=[Path(ref)] if Path(ref).is_absolute() else [p.parent/ref,BASE/ref]
  target=next((c.resolve() for c in candidates if c.is_file()),candidates[0].resolve())
  status='present' if target.is_file() else 'missing_or_template_reference'
  edges.append(dict(source=str(p),target=str(target),raw_reference=ref,kind=kind,status=status))
  if status=='present':queue.append(target)
# Current authoritative artifacts and delivered historical contracts fill code-generated paths.
for p in [BASE/'CONTRACT.json',BASE/'PROVENANCE.json',BASE/'tools/playground.py',BASE/'tools/challenger_template.py',BASE/'data/split_ledger.csv',BASE/'benchmarks/frozen_models.json',*sorted((BASE/'research/original_records').glob('*.json'))]:
 if p.resolve() not in seen:
  raw=p.read_text(encoding='utf-8-sig')
  records.append(dict(path=str(p.resolve()),status='supplementary_authoritative_source_read',bytes=p.stat().st_size,
   relevant_lines=[dict(line=i+1,text=s) for i,s in enumerate(raw.splitlines()) if re.search(r'train|validat|develop|holdout|purge|split|cutoff|test|fold',s,re.I)]))
(OUT/'navigation_audit.json').write_text(json.dumps(dict(entry=str(BASE/'START_HERE.html'),files=records,edges=edges),indent=2),encoding='utf-8')
print('Files read:',len(records),'References:',len(edges))
for r in records:
 print(Path(r['path']).relative_to(BASE) if Path(r['path']).is_relative_to(BASE) else r['path'],r['status'])
print('Missing explicit links:')
for e in edges:
 if e['kind']=='link' and e['status']!='present':print(e['raw_reference'] if 'raw_reference' in e else e['target'])
