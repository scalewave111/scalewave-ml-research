"""Additional contract and model metadata checks on the unmodified package."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'source/SAK_GROSS_V01_COMPLETE'
sys.path.insert(0,str(SOURCE))
from SAK_GROSS_V01 import GrossModel
from catboost import CatBoostClassifier
import lightgbm as lgb

examples=pd.read_parquet(SOURCE/'examples/features.parquet')
expected=pd.read_parquet(SOURCE/'examples/expected_predictions.parquet')
model=GrossModel();actual=model.predict(examples)
pd.testing.assert_frame_equal(actual,expected,check_exact=True)
pd.testing.assert_frame_equal(model.predict(examples[examples.columns[::-1]]),expected,check_exact=True)
records=[{'check':'exact_examples_all_folds','rows':len(examples),'passed':True},
         {'check':'reordered_input_columns','passed':True}]
def reject(name,frame):
    try:model.predict(frame)
    except (ValueError,TypeError):records.append({'check':name,'passed':True})
    else:raise ValueError('Input guard failed: '+name)
reject('missing_feature',examples.drop(columns=model.direction.schema['direction'][0]))
bad=examples.copy();bad.index=bad.index.tz_localize(None);reject('timezone_missing',bad)
reject('duplicate_timestamp',pd.concat([examples.iloc[:1],examples.iloc[:1]]))
reject('unsorted_times',examples.iloc[::-1])
bad=examples.copy();bad.index=bad.index+pd.Timedelta(seconds=1);reject('non_minute_timestamp',bad)
bad=examples.iloc[:1].copy();bad.index=pd.DatetimeIndex(['2025-01-24T00:01:00Z']);reject('h4a_timestamp',bad)
bad=examples.copy();bad.iloc[0,bad.columns.get_loc('magnitude_scale_bps')]=0;reject('zero_magnitude_scale',bad)
bad=examples.copy();bad.iloc[0,0]=np.inf;reject('infinite_feature',bad)
# Public predictor promises native missing-value handling; verify finite output.
missing=examples.copy();missing.loc[:,list(set(model.direction.schema['direction'])|set(model.direction.schema['magnitude']))]=np.nan
result=model.predict(missing)
assert np.isfinite(result[['p_up','predicted_magnitude_bps','signed_score_bps']]).all().all()
records.append({'check':'missing_features_native_handling','passed':True,'note':'All-NaN inputs still produce forecasts. A production feed needs a separate completeness/staleness gate.'})
folds=[]
for fold in model.direction.bundle['folds']:
    direction=CatBoostClassifier();direction.load_model(str(SOURCE/'model_files'/fold['direction_file']))
    magnitude=lgb.Booster(model_file=str(SOURCE/'model_files'/fold['magnitude_file']))
    assert len(direction.feature_names_)==146 and direction.tree_count_==160
    assert magnitude.num_feature()==98 and magnitude.current_iteration()==180
    calibration=json.loads((SOURCE/'model_files'/fold['calibration_file']).read_text())
    issue=pd.Timestamp(fold['valid_from'])
    assert pd.Timestamp(calibration['last_training_label_end'])<issue
    assert pd.Timestamp(calibration['last_calibration_label_end'])<issue-pd.Timedelta(days=30)
    assert pd.Timestamp(fold['latest_magnitude_label_end'])<issue
    assert 0<=calibration['lambda']<=1 and 0<calibration['prior_up']<1
    folds.append({'quarter':fold['quarter'],'direction_features':len(direction.feature_names_),'direction_trees':direction.tree_count_,
                  'magnitude_features':magnitude.num_feature(),'magnitude_trees':magnitude.current_iteration(),'declared_maturity_pass':True})
output={'tests_passed':len(records),'checks':records,'models_checked':14,'folds':folds,
        'scope':'Unmodified frozen model interfaces and declared metadata. Actual training splits checked separately on native reconstructed labels.'}
(ROOT/'ADDITIONAL_RUNTIME_CHECKS.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps(output,indent=2))
