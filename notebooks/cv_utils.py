import numpy as np
from sklearn.metrics import f1_score


def run_cv_f1(clf, cv, X, y, calculate_on_train=True, verbose=True):
    # We create two eampty lists to save the metrics at each fold for train
    # and validation.
    metrics = []
    if calculate_on_train:
        metrics_train = []
    # Loop over the different validation folds
    val_iterable = cv.split(X, y)
    for i, (idx_t, idx_v) in enumerate(val_iterable):
        X_train = X[idx_t]
        y_train = y[idx_t]
        X_val = X[idx_v]
        y_val = y[idx_v]

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        metric = f1_score(y_val, y_pred)
        metrics.append(metric)
        if calculate_on_train:
            y_t_pred = clf.predict(X_train)
            metric_train = f1_score(y_train, y_t_pred)
            metrics_train.append(metric_train)
        if verbose:
            print('{}-fold / {} completed!'.format(i + 1,
                                                   cv.get_n_splits()))
    if calculate_on_train:
        if verbose:
            print('F1 value (Train): {:.2f} ± {:.2f}'.format(
                np.mean(metrics_train),
                np.std(metrics_train, ddof=1)
            ))
            print('F1 value (Val): {:.2f} ± {:.2f}'.format(
                np.mean(metrics),
                np.std(metrics, ddof=1)
            ))
        return metrics, metrics_train
    else:
        if verbose:
            print('F1 value (Val): {:.2f} ± {:.2f}'.format(
                np.mean(metrics),
                np.std(metrics, ddof=1)
            ))
        return metrics
