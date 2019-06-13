import matplotlib.pyplot as plt
import numpy as np

from scipy import interp

from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve


def print_scores_cv(scores, print_timing=False):
    """
    Print the scores returned by sklearn.model_selection.cross_validate.
    Parameters
    ==========
    print_timing : bool, default=False
        If True, print also 'fit_time' and 'score_time'
    Returns
    =======
    None
    """
    for name, vector in scores.items():
        if print_timing:
            print('{}: {:.2f} +- {:.2f}'.format(name, vector.mean(),
                                                vector.std(ddof=1)))
        else:
            if name not in ['fit_time', 'score_time']:
                print('{}: {:.2f} +- {:.2f}'.format(name, vector.mean(),
                                                    vector.std(ddof=1)))
    return None


def run_cv_f1(clf, cv, X, y, calculate_on_train=True, verbose=True):
    """
    Print the scores returned by sklearn.model_selection.cross_validate.
    Parameters
    ----------
    clf : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like
        The data to fit. Can be for example a list, or an array.

    y : array-like
        The target variable to try to predict.

    cv : cross-validation generator or an iterable

    calculate_on_train : bool, default=True
        Calculate metrics on train set

    verbose : bool, default=True
        Print a message at the end of the fold and the
        mean and std of the metric f1_score.

    Returns
    =======
    metrics : np.array
    metrics_train : np.array
        Optional, if calculate_on_train=True.
    """
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


def plot_cv_roc(clf, cv, X, y, figsize=(8, 8)):
    """
    Plots the ROC curve for the cross-validation sets.
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    cv : cross-validation generator
    X : array-like
        The data to fit. Can be for example a list, or an array.
    y : array-like
        The target variable to try to predict in the case of
        supervised learning.
    Returns
    -------
    None
    References
    ----------
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    """
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=figsize)
    i = 0
    for train, test in cv.split(X, y):
        probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_cv_roc_prc(clf, cv, X, y, figsize=(16, 8)):
    """
    Plots the ROC and Precision-Recall curves for the cross-validation sets.
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    cv : cross-validation generator
    X : array-like
        The data to fit. Can be for example a list, or an array.
    y : array-like
        The target variable to try to predict in the case of
        supervised learning.
    Returns
    -------
    None
    References
    ----------
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    """

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    recalls = []
    mean_precs = np.linspace(0, 1, 100)
    aucs2 = []
    plt.figure(figsize=figsize)
    i = 0
    for train, test in cv.split(X, y):
        probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        plt.subplot(121)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        prec, rec, thresholds2 = precision_recall_curve(y[test],
                                                        probas_[:, 1])
        recalls.append(interp(mean_precs, prec, rec))
        recalls[-1][0] = 1.0
        prc_auc = auc(rec, prec)
        aucs2.append(prc_auc)
        plt.subplot(122)
        plt.plot(rec, prec, lw=1, alpha=0.3,
                 label='PRC fold %d (AUC = %0.2f)' % (i, prc_auc))

        i += 1
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    mean_rec = np.mean(recalls, axis=0)
    mean_rec[-1] = 0.0
    mean_auc2 = auc(mean_rec, mean_precs)
    std_auc2 = np.std(aucs2)
    std_rec = np.std(recalls, axis=0)
    recs_upper = np.minimum(mean_rec + std_rec, 1)
    recs_lower = np.maximum(mean_rec - std_rec, 0)

    plt.subplot(121)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    plt.subplot(122)
    plt.plot([1, 0], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    plt.plot(mean_rec, mean_precs, color='b',
             label=r'Mean PRC (AUC = %0.2f $\pm$ %0.2f)' % (
                 mean_auc2, std_auc2),
             lw=2, alpha=.8)

    plt.fill_betweenx(mean_precs, recs_lower, recs_upper, color='grey',
                      alpha=.2, label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend(loc="lower left")

    plt.show()
