"""Copy into your member folder and implement fit_predict. Never fit on later labels."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from playground import load_training, load_development_inputs, training_folds

def fit_predict(X_train, y_train, X_development):
    # Choose features/hyperparameters using training_folds() only.
    # Fit imputation, scaling and feature admission separately inside every fold.
    # Fit your frozen final procedure on all allowed training rows.
    # Return a DataFrame indexed exactly like X_development, with columns:
    # magnitude_log_bps (nonnegative), p_up (within [0,1]).
    # If working on one track, use the frozen model's predictions for the other.
    raise NotImplementedError('Implement your registered hypothesis; no default model search is launched.')

if __name__=='__main__':
    X,y=load_training();dev=load_development_inputs()
    result=fit_predict(X,y,dev)
    result.to_csv('candidate_predictions.csv',index_label='decision_time',float_format='%.17g')
