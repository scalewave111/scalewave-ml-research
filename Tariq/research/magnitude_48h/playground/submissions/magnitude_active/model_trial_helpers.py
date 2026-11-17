"""MAE models for the controlled local comparison; no work on import."""
import copy
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

SEEDS=[48,17,91]
SETTINGS={'hgb':[{'leaves':7},{'leaves':15}],
          'nn':[{'widths':[32,16]},{'widths':[64,32]}]}

def network(dim,widths):
 return torch.nn.Sequential(torch.nn.Linear(dim,widths[0]),torch.nn.ReLU(),
  torch.nn.Linear(widths[0],widths[1]),torch.nn.ReLU(),
  torch.nn.Linear(widths[1],1),torch.nn.Softplus())

def predict(bundle,X):
 if bundle['family']=='hgb':
  with threadpool_limits(limits=8):p=bundle['model'].predict(X)
 else:
  model=network(X.shape[1],bundle['settings']['widths'])
  model.load_state_dict(bundle['state']);model.eval()
  with torch.no_grad():p=model(torch.tensor(X,dtype=torch.float32)).squeeze(1).numpy()*bundle['scale']
 return np.maximum(p,0).astype(float)

def fit(family,settings,seed,A,y,B=None,vy=None,epochs=None,max_epochs=80):
 """Only inner fitting receives validation labels; outer refits receive fixed epochs."""
 torch.set_num_threads(8)
 if family=='hgb':
  model=HistGradientBoostingRegressor(loss='absolute_error',learning_rate=.05,
   max_iter=200,max_leaf_nodes=settings['leaves'],min_samples_leaf=64,
   l2_regularization=1.,early_stopping=False,random_state=seed)
  with threadpool_limits(limits=8):model.fit(A,y)
  return {'family':family,'settings':settings,'seed':seed,'model':model}
 assert (B is None)==(vy is None)
 assert (epochs is not None) if B is None else (epochs is None)
 torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
 model=network(A.shape[1],settings['widths']);scale=max(float(np.median(y)),1.)
 x=torch.tensor(A,dtype=torch.float32);t=torch.tensor(y/scale,dtype=torch.float32)
 if B is not None:bx=torch.tensor(B,dtype=torch.float32)
 opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.001)
 generator=torch.Generator().manual_seed(seed)
 best=float('inf');best_state=None;best_epoch=0;stale=0;trace=[]
 for epoch in range(1,(epochs if epochs is not None else max_epochs)+1):
  model.train()
  for ix in torch.randperm(len(x),generator=generator).split(512):
   opt.zero_grad();loss=torch.nn.functional.l1_loss(model(x[ix]).squeeze(1),t[ix])
   assert torch.isfinite(loss);loss.backward();opt.step()
  if B is not None:
   model.eval()
   with torch.no_grad():err=float(np.abs(model(bx).squeeze(1).numpy()*scale-vy).mean())
   assert np.isfinite(err);trace.append(err)
   if err<best:
    best=err;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
   else:stale+=1
   if stale>=10:break
  if epoch%10==0:print(f'    NN epoch {epoch}',flush=True)
 if B is None:best_state=copy.deepcopy(model.state_dict());best_epoch=epochs
 return {'family':family,'settings':settings,'seed':seed,'state':best_state,
  'scale':scale,'epochs':best_epoch,'epochs_run':epoch,'validation_MAE':trace}

def choose(inner_predictions,y,pair):
 rows=[]
 for setting,p in enumerate(inner_predictions):
  for weight in [0.,.1,.25,.5,1.]:
   rows.append({'setting':setting,'weight':weight,'MAE':float(np.abs(y-((1-weight)*pair+weight*p)).mean())})
 standalone=min((r for r in rows if r['weight']==1.),key=lambda r:(r['MAE'],r['setting']))
 blended=min((r for r in rows if r['weight']<1.),key=lambda r:(r['MAE'],r['weight'],r['setting']))
 return {'standalone':standalone,'blend':blended},rows
