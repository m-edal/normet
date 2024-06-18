import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.base import clone

def pdp_all(automl, df, feature_names=None,variables=None, training_only=True,n_cores = -1):
    """
    Computes partial dependence plots for all specified features.

    Parameters:
        automl: AutoML model object.
        df (DataFrame): Input DataFrame containing the dataset.
        feature_names (list): List of feature names to compute partial dependence plots for.
        variables (list, optional): List of variables to compute partial dependence plots for. If None, defaults to feature_names.
        training_only (bool, optional): If True, computes partial dependence plots only for the training set. Default is True.
        n_cores (int, optional): Number of CPU cores to use for parallel computation. Default is -1 (uses all available cores).

    Returns:
        DataFrame: DataFrame containing the computed partial dependence plots for all specified features.
    """
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
    """
    Worker function for computing partial dependence plots for a single feature.

    Parameters:
        automl: AutoML model object.
        X_train (DataFrame): Input DataFrame containing the training data.
        variable (str): Name of the feature to compute partial dependence plot for.
        training_only (bool, optional): If True, computes partial dependence plot only for the training set. Default is True.

    Returns:
        DataFrame: DataFrame containing the computed partial dependence plot for the specified feature.
    """
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


def pdp_plot(automl,df,feature_names,variables=None,kind='average',n_cores=-1,training_only=True,figsize=(8,8),hspace=0.5):
    """
    Plots partial dependence plots for specified features.

    Parameters:
        automl: AutoML model object.
        df (DataFrame): Input DataFrame containing the dataset.
        feature_names (list): List of feature names to plot partial dependence plots for.
        variables (list, optional): List of variables to plot partial dependence plots for. If None, defaults to feature_names.
        kind (str, optional): Type of plot to generate. Default is 'average'.
        n_cores (int, optional): Number of CPU cores to use for parallel computation. Default is -1 (uses all available cores).
        training_only (bool, optional): If True, plots partial dependence plots only for the training set. Default is True.
        figsize (tuple, optional): Size of the figure. Default is (8, 8).
        hspace (float, optional): Height space between subplots. Default is 0.5.

    Returns:
        PartialDependenceDisplay: Partial dependence plot display object.
    """
    if variables is None:
        variables = feature_names

    if training_only:
        df = df[df["set"] == "training"]
    X_train, y_train = df[feature_names], df['value']
    fig, ax = plt.subplots(figsize=figsize)
    result = PartialDependenceDisplay.from_estimator(automl, X_train, variables,kind=kind,n_jobs=n_cores,ax=ax)
    plt.subplots_adjust(hspace=hspace)
    return result


def pdp_interaction(automl,df,variables,kind='average',training_only=True,ncols=3,figsize=(8,4),constrained_layout=True):
    """
    Plots interaction partial dependence plots for specified features.

    Parameters:
        automl: AutoML model object.
        df (DataFrame): Input DataFrame containing the dataset.
        variables (list): List of feature names to plot interaction partial dependence plots for.
        kind (str, optional): Type of plot to generate. Default is 'average'.
        training_only (bool, optional): If True, plots interaction partial dependence plots only for the training set. Default is True.
        ncols (int, optional): Number of columns for subplots. Default is 3.
        figsize (tuple, optional): Size of the figure. Default is (8, 4).
        constrained_layout (bool, optional): If True, adjusts subplots to fit into the figure area. Default is True.

    Returns:
        PartialDependenceDisplay: Interaction partial dependence plot display object.
    """
    if training_only:
        df = df[df["set"] == "training"]
    fig, ax = plt.subplots(ncols=ncols, figsize=figsize, constrained_layout=constrained_layout)
    result = PartialDependenceDisplay.from_estimator(automl, df, features=variables,kind=kind,ax=ax)
    return result


def pdp_nointeraction(automl,df,feature_names,variables=None,kind='average',training_only=True,ncols=3,figsize=(8,4),constrained_layout=True):
    """
    Plots partial dependence plots without interaction effects for specified features.

    Parameters:
        automl: AutoML model object.
        df (DataFrame): Input DataFrame containing the dataset.
        feature_names (list): List of feature names to plot partial dependence plots for.
        variables (list, optional): List of variables to plot partial dependence plots for. If None, defaults to feature_names.
        kind (str, optional): Type of plot to generate. Default is 'average'.
        training_only (bool, optional): If True, plots partial dependence plots only for the training set. Default is True.
        ncols (int, optional): Number of columns for subplots. Default is 3.
        figsize (tuple, optional): Size of the figure. Default is (8, 4).
        constrained_layout (bool, optional): If True, adjusts subplots to fit into the figure area. Default is True.

    Returns:
        PartialDependenceDisplay: Partial dependence plot display object.
    """
    if training_only:
        df = df[df["set"] == "training"]
    #factorize category variables
    category_cols = df.select_dtypes(['category']).columns.tolist()
    for i,cat_col in enumerate(category_cols):
        codes, uniques = pd.factorize(df[cat_col])
        df.loc[:,cat_col]=codes

    X_train, y_train = df[feature_names], df['value']
    interaction_cst = [[name] for name in feature_names]

    model_without_interactions = (
        clone(automl.model.estimator)
        .set_params(interaction_constraints = interaction_cst)
        .fit(X_train, y_train))
    fig, ax = plt.subplots(ncols=ncols, figsize=figsize, constrained_layout=constrained_layout)
    result = PartialDependenceDisplay.from_estimator(model_without_interactions, X_train, features=variables,kind=kind,ax=ax)
    return result
