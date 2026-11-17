"""Replay saved fine-tuned candidate without fitting: python replay.py X.parquet output.csv."""
import sys
from pathlib import Path
import run as study
if __name__=='__main__':
    assert len(sys.argv)==3,'Usage: replay.py INPUT.parquet OUTPUT.csv'
    dest=Path(sys.argv[2]);assert not dest.exists(),'Refusing to overwrite existing output'
    study.verify();X=study.pd.read_parquet(sys.argv[1]);assert X.index.is_unique
    models=study.load('final_pipeline.json')['models']
    out=study.b.frozen_predictions(X)
    out['p_up']=study.np.mean([study.predict(X,m) for m in models],axis=0)
    out.to_csv(dest,index_label='decision_time',float_format='%.17g')
    print('Saved',len(out),'predictions to',dest)
