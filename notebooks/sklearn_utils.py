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
from sklearn.preprocessing import FunctionTransformer


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
