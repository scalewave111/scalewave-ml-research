"""Bounded net300h MAE refinement; original backend files remain unchanged."""
import mae_both_300h as legacy
b=legacy.b;tree=legacy.tree;neural=legacy.neural
PACKS=['leverage'];WEIGHTS=[.05,.1,.25,.4,.5,.6,.75,.9,1.];ANCHOR_WEIGHTS=[0.,.25,.5,.75,1.]
CAP=legacy.CAP;EPOCHS=legacy.EPOCHS;LAGS=legacy.LAGS
pack_columns=legacy.pack_columns;augment=legacy.augment;windows=legacy.windows;predict=legacy.predict
neural.CONFIG['gru16_regularized']=dict(neural.CONFIG['gru16'],decay=.1)
CONFIG=dict(legacy.CONFIG)
CONFIG['hgb_shallow_l1']=dict(CONFIG['hgb_l1'],max_depth=1,max_leaf_nodes=2)
CONFIG['gru16_regularized_l1']=dict(CONFIG['gru16_l1'],base='gru16_regularized',decay=.1)
_original_fit=legacy.fit

def fit(x,y,spec):
    if spec['model']=='median':return _original_fit(x,y,spec)
    cfg=CONFIG[spec['config']]
    previous=tree.CONFIG.copy()
    try:
        if cfg['base']=='hgb':
            tree.CONFIG.update({k:cfg[k] for k in previous if k in cfg})
        return _original_fit(x,y,spec)
    finally:
        tree.CONFIG.clear();tree.CONFIG.update(previous)

def search(x,y,folds,folder):return legacy.search(x,y,folds,folder)
# Only this new process is configured; no source files or old artifacts edited.
tree.PACKS=PACKS;legacy.PACKS=PACKS;legacy.CONFIG=CONFIG;legacy.WEIGHTS=WEIGHTS;legacy.ANCHOR_WEIGHTS=ANCHOR_WEIGHTS;legacy.fit=fit
