from sklearn import metrics
import numpy as np


def print_metrics_binary_classification(y_true, prediction_probs, setting, verbose=1, logger=None, wandb=None):
    if sum(y_true) != 0.:
        logger.info("Number of labeled examples: {}".format(y_true.shape[0]))
        logger.info("Number of examples with mortality 1: {}".format(sum(y_true == 1.)))
    else:
        logger.info("Warning: Couldn't compute AUC -- all examples are from the same class")
        return

    prediction_probs = np.array(prediction_probs)
    
    # calculate roc curves
    fpr, tpr, thresholds = metrics.roc_curve(y_true, prediction_probs)
    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    logger.info('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

    predictions = (prediction_probs >= thresholds[ix]).astype(int)

    # # cf = metrics.confusion_matrix(y_true, predictions, labels=range(2))
    # # if verbose:
    # #     logger.info('Confusion matrix:')
    # #     logger.info(cf)

    acc = metrics.accuracy_score(y_true, predictions)
    precision = metrics.precision_score(y_true, predictions)
    recall = metrics.recall_score(y_true, predictions)
    f1macro = metrics.f1_score(y_true, predictions, average='macro')

    auroc = metrics.roc_auc_score(y_true, prediction_probs)
    auprc = metrics.average_precision_score(y_true, prediction_probs)
    
    results = {
               setting +' AUROC': auroc,
               setting +' AUPRC': auprc,
               setting +' Accuracy': acc,
               setting +' Precision': precision,
               setting +' Recall': recall,
               setting +' F1-score': f1macro
               }
        
    if verbose:
        for key in results:
            logger.info('{} = {:.4f}'.format(key, results[key]))
    if wandb is not None:
        wandb.log(results)

    return results