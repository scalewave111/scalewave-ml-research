from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'source/SAK_GROSS_V01_COMPLETE'
prices=pd.read_parquet(Path(r'D:\Team\November\HANDOFF_4_ML_RESEARCH_V1\data\prices\PRICE_1m_BTCUSDT.parquet'),columns=['decision_time','f066__value','f066__is_valid']).set_index('decision_time')
rows=[]
for file in sorted((SOURCE/'evidence/replay').glob('20*.parquet')):
    frame=pd.read_parquet(file,columns=['actual_bps'])
    entry=prices.reindex(frame.index+pd.Timedelta(minutes=1));exit=prices.reindex(frame.index+pd.Timedelta(minutes=16))
    actual=10000*(exit.f066__value.to_numpy('float64')/entry.f066__value.to_numpy('float64')-1)
    error=float(np.max(np.abs(actual-frame.actual_bps.to_numpy())))
    assert np.isfinite(actual).all() and error<1e-8 and entry.f066__is_valid.all() and exit.f066__is_valid.all()
    rows.append({'quarter':file.stem,'rows':len(frame),'max_price_formula_error_bps':error})
result={'status':'PASS','formula':'10000*(native P(t+16)/native P(t+1)-1)','rows':sum(x['rows'] for x in rows),'quarters':rows,'scope':'Float64 endpoint formula check on every replay origin; native full-path validity is checked with native Y separately'}
(ROOT/'PRICE_TARGET_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('Price target checks passed; max error bps:',max(x['max_price_formula_error_bps'] for x in rows))
