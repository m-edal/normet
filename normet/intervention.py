import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

def scm_parallel(df, poll_col, date_col, code_col, control_pool, post_col, n_cores = -1):
    """
    Performs Synthetic Control Method (SCM) in parallel for multiple treatment targets.

    Parameters:
        df (DataFrame): Input DataFrame containing the dataset.
        poll_col (str): Name of the column containing the poll data.
        date_col (str): Name of the column containing the date data.
        code_col (str): Name of the column containing the code data.
        control_pool (list): List of control pool codes.
        post_col (str): Name of the column indicating the post-treatment period.
        n_cores (int, optional): Number of CPU cores to use. Default is -1 (uses all available cores).

    Returns:
        DataFrame: DataFrame containing synthetic control results for all treatment targets.
    """
    treatment_pool = df[code_col].unique()
    synthetic_all = pd.concat(Parallel(n_jobs=n_cores)(delayed(scm)(
                    df=df,
                    poll_col=poll_col,
                    date_col=date_col,
                    code_col=code_col,
                    treat_target=code,
                    control_pool=control_pool,
                    post_col=post_col) for code in treatment_pool))
    return synthetic_all


def scm(df, poll_col, date_col, code_col, treat_target, control_pool, post_col):
    """
    Performs Synthetic Control Method (SCM) for a single treatment target.

    Parameters:
        df (DataFrame): Input DataFrame containing the dataset.
        poll_col (str): Name of the column containing the poll data.
        date_col (str): Name of the column containing the date data.
        code_col (str): Name of the column containing the code data.
        treat_target (str): Code of the treatment target.
        control_pool (list): List of control pool codes.
        post_col (str): Name of the column indicating the post-treatment period.

    Returns:
        DataFrame: DataFrame containing synthetic control results for the specified treatment target.
    """
    x_pre_control = (df[(df[code_col]!=treat_target)&(df[code_col].isin(control_pool))]
                     .query(f"~{post_col}")
                     .pivot_table(index=date_col, columns=code_col, values=poll_col)
                     .values)

    y_pre_treat_mean = (df
                        .query(f"~{post_col}")[df[code_col]==treat_target].groupby(date_col)
                        [poll_col]
                        .mean())
    param_grid = {'alpha': [i/10 for i in range(1, 101)]}
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=5)
    grid_search.fit(x_pre_control, y_pre_treat_mean.values.reshape(-1, 1))
    best_alpha = grid_search.best_params_['alpha']
    ridge_final = Ridge(alpha=best_alpha,fit_intercept=False)
    ridge_final.fit(x_pre_control, y_pre_treat_mean.values.reshape(-1, 1))
    w = ridge_final.coef_.flatten()
    sc = (df[(df[code_col]!=treat_target)&(df[code_col].isin(control_pool))]
      .pivot_table(index=date_col, columns=code_col, values=poll_col)
      .values) @ w

    data=(df
            [df[code_col]==treat_target][[date_col, code_col, poll_col]]
            .assign(synthetic=sc)).set_index(date_col)
    data['effects']=data[poll_col]-data['synthetic']
    return data


def ml_syn(df, poll_col, date_col, code_col, treat_target, control_pool, cutoff_date,training_time=60):
    """
    Performs synthetic control using machine learning regression models.

    Parameters:
        df (DataFrame): Input DataFrame containing the dataset.
        poll_col (str): Name of the column containing the poll data.
        date_col (str): Name of the column containing the date data.
        code_col (str): Name of the column containing the code data.
        treat_target (str): Code of the treatment target.
        control_pool (list): List of control pool codes.
        cutoff_date (str): Date for splitting pre- and post-treatment datasets.
        training_time (int, optional): Total running time in seconds for the AutoML model. Default is 60.

    Returns:
        DataFrame: DataFrame containing synthetic control results for the specified treatment target.
    """
    from flaml import AutoML
    automl = AutoML()
    from sklearn.metrics import r2_score
    dfp=(df[df[code_col].isin(control_pool+[treat_target])]).pivot_table(index=date_col, columns=code_col, values=poll_col)
    pre_dataset=dfp[dfp.index<cutoff_date]
    post_dataset=dfp[dfp.index>=cutoff_date]
    settings = {
    "time_budget": training_time,  # total running time in seconds
    "metric": "r2",  # primary metric
    "task": "regression",  # task type
    "eval_method": "cv",
    'seed':987654321}
    automl.fit(dataframe=pre_dataset, label=treat_target,**settings)
    pre_pred=automl.predict(pre_dataset)

    data=(df
        [df[code_col]==treat_target][[date_col, code_col, poll_col]]
        .assign(synthetic=automl.predict(dfp))).set_index(date_col)
    data['effects']=data[poll_col]-data['synthetic']
    return data


def ml_syn_parallel(df, poll_col, date_col, code_col, control_pool, cutoff_date,training_time=60, n_cores = -1):
    """
    Performs synthetic control using machine learning regression models in parallel for multiple treatment targets.

    Parameters:
        df (DataFrame): Input DataFrame containing the dataset.
        poll_col (str): Name of the column containing the poll data.
        date_col (str): Name of the column containing the date data.
        code_col (str): Name of the column containing the code data.
        control_pool (list): List of control pool codes.
        cutoff_date (str): Date for splitting pre- and post-treatment datasets.
        training_time (int, optional): Total running time in seconds for the AutoML model. Default is 60.
        n_cores (int, optional): Number of CPU cores to use. Default is -1 (uses all available cores).

    Returns:
        DataFrame: DataFrame containing synthetic control results for all treatment targets.
    """
    treatment_pool = df[code_col].unique()
    synthetic_all = pd.concat(Parallel(n_jobs=n_cores)(delayed(ml_syn)(
                    df=df,
                    poll_col=poll_col,
                    date_col=date_col,
                    code_col=code_col,
                    treat_target=code,
                    control_pool=control_pool,
                    cutoff_date=cutoff_date,
                    training_time=training_time) for code in treatment_pool))
    return synthetic_all
