"""Small synthetic checks only; no company-data fitting and no artifacts."""
import io
import joblib,numpy as np,torch
from model_trial_helpers import fit,predict,choose,network

def roundtrip(model):
 stream=io.BytesIO();joblib.dump(model,stream);stream.seek(0);return joblib.load(stream)

rng=np.random.default_rng(17);A=rng.normal(size=(512,4));B=rng.normal(size=(100,4))
y=25+3*A[:,0];vy=25+3*B[:,0]
model=fit('hgb',{'leaves':7},48,A,y)
assert model['model'].loss=='absolute_error' and model['model'].early_stopping is False
np.testing.assert_array_equal(predict(model,B),predict(roundtrip(model),B))
assert np.abs(predict(model,B)-vy).mean()<np.abs(np.median(y)-vy).mean()
settings={'widths':[32,16]}
torch.manual_seed(48);initial=network(4,settings['widths'])
with torch.no_grad():initial_error=np.abs(initial(torch.tensor(B,dtype=torch.float32)).squeeze(1).numpy()*np.median(y)-vy).mean()
model=fit('nn',settings,48,A,y,B,vy,max_epochs=20)
p=predict(model,B);assert np.isfinite(p).all() and (p>=0).all()
assert abs(np.abs(p-vy).mean()-min(model['validation_MAE']))<1e-5
assert np.abs(p-vy).mean()<initial_error
np.testing.assert_array_equal(p,predict(roundtrip(model),B))
outer=fit('nn',settings,48,A,y,epochs=model['epochs'])
assert outer['epochs_run']==model['epochs'] and outer['validation_MAE']==[]
outer2=fit('nn',settings,48,A,y,epochs=model['epochs'])
np.testing.assert_array_equal(predict(outer,B),predict(outer2,B))
sel,rows=choose([np.array([12.,12.]),np.array([20.,20.])],np.array([10.,10.]),np.array([8.,8.]))
assert sel['standalone']['setting']==0
assert sel['blend']['setting']==0 and sel['blend']['weight']==.5 and sel['blend']['MAE']==0
print('PASS: MAE learners, deterministic seeds, selected epoch restoration, fixed-epoch outer refit, serialization replay, and earlier-validation setting/blend selection. No real-data training performed.')
