from sklearn import metrics
import numpy as np


def print_metrics_binary_classification(y_true, prediction_probs, setting, verbose=1, logger=None, wandb=None):
    
    prediction_probs = np.array(prediction_probs)

    auroc = metrics.roc_auc_score(y_true, prediction_probs)
    
    results = {
               setting +' AUROC': auroc
               }
        
    if verbose:
        for key in results:
            logger.info('{} = {:.4f}'.format(key, results[key]))
    
    if wandb is not None:
        wandb.log(results)
    
    return results

