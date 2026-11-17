from pathlib import Path
import sys, re, json
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'Dataset/TEAM_48H_PLAYGROUND_V1/TEAM_48H_PLAYGROUND_V1'
sys.path.insert(0,str(PACKAGE/'submissions/member_02'))
from parquet_compat import read_frame
from pypdf import PdfReader
source = next((Path.home()/'Downloads').glob('Top 20*MAGNITUDE*.pdf'))
reader = PdfReader(source)
# Pages 1-2: all-horizon magnitude; pages 5-8: magnitude section after direction.
first = '\n'.join(p.extract_text() or '' for p in reader.pages[:2])
later = '\n'.join(p.extract_text() or '' for p in reader.pages[4:])
later = later[later.index('MAGNITUDE'):]
pattern = r'\b(?:flow|price|basis|volatility|liquidation|funding|oi)_[a-z0-9_]+__[a-z0-9_]+'
names = list(dict.fromkeys(re.findall(pattern, first+'\n'+later)))
x = read_frame(PACKAGE/'data/X_train.parquet')
canonical = {c.removeprefix('F4Z__').removesuffix('__value').replace('__','_',1):c for c in x}
mapping = [{'report_feature':n,'column':canonical.get(n),'mapped':n in canonical} for n in names]
out = ROOT/'Report/ic_rf_sources'
pd.DataFrame(mapping).to_csv(out/'magnitude_feature_mapping.csv',index=False)
(out/'magnitude_feature_mapping.json').write_text(json.dumps(mapping,indent=2))
print(json.dumps(mapping,indent=2))
print('Training shape:',x.shape,'date range:',x.index.min(),x.index.max())
print('Representative columns:',list(x)[:5])
print('Magnitude table 1 opening:',first[:4200].encode('ascii','replace').decode())
