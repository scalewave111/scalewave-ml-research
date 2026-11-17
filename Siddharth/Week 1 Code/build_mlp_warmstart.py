from pathlib import Path

source=Path('mlp_two_liquidation.py').read_text()
source=source.replace('mlp_two_liquidation_v1','mlp_two_liquidation_warmstart_v2')
source=source.replace('BTC_48h_MLP_Two_Liquidation.ipynb','BTC_48h_MLP_Two_Liquidation_Warmstart.ipynb')
source=source.replace("ARCHS={'h8':[8],'h16':[16],'h8_4':[8,4],'h16_8':[16,8]}","ARCHS={'h8_4_slow':[8,4],'h8_4_fast':[8,4],'h16_8_slow':[16,8],'h16_8_fast':[16,8]}\nLINEAR_CACHE={}")
source=source.replace('MAX_EPOCHS=48','MAX_EPOCHS=128')
start=source.index('class Net(nn.Module):')
end=source.index('\ndef design(',start)
source=source[:start]+'''class Net(nn.Module):
 def __init__(self,widths,median):
  super().__init__()
  self.first=nn.Linear(2,widths[0]);self.second=nn.Linear(widths[0],widths[1]);self.head=nn.Linear(widths[1],1)
 def forward(self,x):return self.head(torch.relu(self.second(torch.relu(self.first(x))))).squeeze(1).clamp(min=0)
 def warm_start(self,coef,intercept):
  with torch.no_grad():
   self.first.weight[:4].copy_(torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]))
   self.first.bias[:4].zero_();self.second.weight[:4].zero_();self.second.bias[:4].zero_()
   self.second.weight[:4,:4].copy_(torch.eye(4))
   self.head.weight.zero_();self.head.weight[0,:4].copy_(torch.tensor([coef[0],-coef[0],coef[1],-coef[1]],dtype=torch.float32))
   self.head.bias.fill_(float(intercept))

def initial_linear(tx,ty,cutoff):
 from sklearn.linear_model import QuantileRegressor
 key=str(cutoff)
 if key not in LINEAR_CACHE:
  # Exactly two columns; no flags, baseline predictions, or other regressors.
  linear=QuantileRegressor(quantile=.5,alpha=.0001,solver='highs').fit(tx.numpy().astype(float),ty.numpy().astype(float))
  LINEAR_CACHE[key]=(linear.coef_,float(linear.intercept_))
 return LINEAR_CACHE[key]
''' +source[end:]
source=source.replace("decay=[v for v in model.parameters() if v.ndim>1]", "coef,intercept=initial_linear(tx,ty,cutoff);model.warm_start(coef,intercept)\n np.testing.assert_allclose(infer(model,tx),np.maximum(0,tx.numpy()@coef+intercept)*100,atol=1e-4,rtol=1e-6)\n decay=[v for v in model.parameters() if v.ndim>1]")
source=source.replace("'weight_decay':.01", "'weight_decay':.001")
source=source.replace("lr=.001)", "lr=.0001 if name.endswith('slow') else .001)")
source=source.replace("best=np.inf;stale=0;hist=[];start=time.monotonic()", "best=float(np.mean(abs(infer(model,vx)-vy))) if vx is not None else np.inf\n stale=0;hist=[dict(epoch=0,train_mae=float(np.mean(abs(infer(model,tx)-target))),validation_mae=best if vx is not None else None)]\n state=copy.deepcopy(model.state_dict());best_epoch=0;start=time.monotonic()")
source=source.replace('(epochs or MAX_EPOCHS)', '(MAX_EPOCHS if epochs is None else epochs)')
source=source.replace('stale>=8','stale>=12').replace('early_stopping_patience=8','early_stopping_patience=12')
source=source.replace("optimizer='AdamW lr .001, weight decay .01 on weights only, MAE loss, gradient clip 1'", "optimizer='AdamW lr .0001 or .001, weight decay .001 on weights only, MAE loss, gradient clip 1; include initialization epoch 0'",1)
source=source.replace("dropout=.05,activation='SiLU'", "dropout=0,activation='ReLU',initialization='Two-input median regression embedded exactly in MLP weights; alpha .0001 on target/100; initialization uses training data only'")
source=source.replace("'torch','numpy','pandas','scipy','joblib'", "'torch','numpy','pandas','scipy','joblib','scikit-learn'")
source=source.replace('Two liquidation input MLP: executed', 'Two liquidation input warm-start MLP: executed')
source=source.replace('Training source: mlp_two_liquidation.py.', 'Training source: mlp_two_liquidation_warmstart.py.')
source=source.replace('# Two-input liquidation MLP results', '# Two-input liquidation warm-start MLP results')
Path('mlp_two_liquidation_warmstart.py').write_text(source,encoding='utf-8')
print('Created separate warm-start runner; v1 preserved')
