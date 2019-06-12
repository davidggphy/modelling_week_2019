import numpy as np

def _F1_eval(preds, labels):
    t = np.arange(0, 1, 0.005)
    f = np.repeat(0, 200)
    results = np.vstack([t, f]).T
    # assuming labels only containing 0's and 1's
    n_pos_examples = sum(labels)
    if n_pos_examples == 0:
        raise ValueError("labels not containing positive examples")

    for i in range(200):
        pred_indexes = (preds >= results[i, 0])
        TP = sum(labels[pred_indexes])
        FP = len(labels[pred_indexes]) - TP
        precision = 0
        recall = TP / n_pos_examples

        if (FP + TP) > 0:
            precision = TP / (FP + TP)

        if (precision + recall > 0):
            F1 = 2 * precision * recall / (precision + recall)
        else:
            F1 = 0
        results[i, 1] = F1
    return (max(results[:, 1]))

# Xgboost version
def F1_eval(preds, dtrain):
    f1 = _F1_eval(preds, dtrain.get_label())
    # xgboost minimizes metrics
    return 'f1_err', 1.-f1