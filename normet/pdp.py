import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.base import clone

def pdp_all(automl, df, feature_names=None,variables=None, training_only=True):
    # 使用joblib库进行并行计算
    n_cores = -1  # 使用所有可用CPU核心
    if variables is None:
        variables = feature_names
    if training_only:
        df = df[df["set"] == "training"]
    X_train, y_train = df[feature_names], df['value']

    results = Parallel(n_jobs=n_cores)(delayed(pdp_worker)(automl,X_train,var) for var in variables)
    df_predict = pd.concat(results)
    df_predict.reset_index(drop=True, inplace=True)
    return df_predict


def pdp_worker(automl, X_train, variable,training_only=True):
    # Filter only to training set
    results = partial_dependence(estimator=automl, X=X_train,
                                 features=variable,kind='individual')

    # Alter names and add variable
    df_predict = pd.DataFrame({"value": results['values'][0],
                                "pdp_mean": np.mean(results['individual'][0],axis=0),
                               'pdp_std':np.std(results['individual'][0],axis=0)})
    df_predict["variable"] = variable
    df_predict = df_predict[["variable", "value", "pdp_mean","pdp_std"]]

    return df_predict


def pdp_plot(automl,df,feature_names,variables=None,kind='average',training_only=True,figsize=(8,8),hspace=0.5,n_jobs=-1):
    if variables is None:
        variables = feature_names

    if training_only:
        df = df[df["set"] == "training"]
    X_train, y_train = df[feature_names], df['value']
    fig, ax = plt.subplots(figsize=figsize)
    result = PartialDependenceDisplay.from_estimator(automl, X_train, variables,kind=kind,n_jobs=-1,ax=ax)
    plt.subplots_adjust(hspace=hspace)
    return result


def pdp_interaction(automl,df,variables,kind='average',training_only=True,ncols=3,figsize=(8,4),constrained_layout=True):

    if training_only:
        df = df[df["set"] == "training"]
    fig, ax = plt.subplots(ncols=ncols, figsize=figsize, constrained_layout=constrained_layout)
    result = PartialDependenceDisplay.from_estimator(automl, df, features=variables,kind=kind,ax=ax)
    return result


def pdp_nointeraction(automl,df,feature_names,variables=None,kind='average',training_only=True,ncols=3,figsize=(8,4),constrained_layout=True):
    if training_only:
        df = df[df["set"] == "training"]
    X_train, y_train = df[feature_names], df['value']
    interaction_cst = [[i] for i in range(X_train.shape[1])]
    model_without_interactions = (
        clone(automl.model.estimator)
        .set_params(interaction_constraints = interaction_cst)
        .fit(X_train, y_train))
    fig, ax = plt.subplots(ncols=ncols, figsize=figsize, constrained_layout=constrained_layout)
    result = PartialDependenceDisplay.from_estimator(model_without_interactions, X_train, features=variables,kind=kind,ax=ax)
    return result
