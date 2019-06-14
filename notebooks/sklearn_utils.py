"""
Transformer function to select a subset of features.
It can be easily included into a sklearn's Pipeline.

Example of use
==============
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

list_features_names = ['V1','V2']
feat_select = FunctionTransformer(_select_features_dic,validate=True,
                    kw_args={'list_features_names_names':list_features_names,
                             'feat_to_idx':feat_to_idx})
clf_ = LogisticRegression()
clf = make_pipeline(feat_select, clf_)

"""
import numpy as np

from sklearn.preprocessing import FunctionTransformer


def hard_vote_predict(estimators, X, weights=None):
    """
    Combine a dictionary of estimators to create a hard voting ensemble.
    Parameters
    ----------
    estimators : dict
        Dictionary with name (str): model entries with predict method.
        If the method predict returns probabilities, then the name should
        end with 'prob'.
    X : np.array
        Input.
    weights : list, tuple or np.array, default=None
        List of weights for each estimator. If None, then it is uniform.
    """
    if weights is None:
        weights = np.ones(len(estimators))
    else:
        assert len(weights) == len(
            estimators), 'Number of estimators should be the same as number of weights'
        weights = np.array(weights)
    weights = weights.reshape((-1, 1))
    y_preds = []
    for name, clf in estimators.items():
        y_pred = clf.predict(X)
        if name.endswith('prob'):
            y_pred = (1 * (y_pred > 0.5)).reshape((-1))
        y_preds.append(y_pred)

    y_preds = np.array(y_preds)
    y_final = 1 * (np.mean(weights * y_preds, axis=0) > 0.5)
    return y_final


def FeatureSelector(list_features_idx):
    """
    Constructs a transformer that selects a subset of the features
    given their indices list_features_idx in the X array.
    """
    return FunctionTransformer(_select_features, validate=True,
                               kw_args={'list_idx': list_features_idx})


def FeatureSelectorDic(list_features_names, feat_to_idx):
    """
    Constructs a transformer that selects a subset of the features
    given the name of the features list_features_names and a dictionary
    feat_to_idx associating each feature with the index in the X array.
    """
    return FunctionTransformer(_select_features_dic, validate=True,
                               kw_args={
                                   'list_features_names': list_features_names,
                                   'feat_to_idx': feat_to_idx})


def _select_features(X, list_idx):
    return X[:, list_idx]


def _select_features_dic(X, list_features_names, feat_to_idx):
    list_idx = [feat_to_idx[feat] for feat in list_features_names]
    return X[:, list_idx]
