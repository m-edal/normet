import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from pdpbox import pdp

def partial_dependencies(automl, df, feature_names=None,variable=None, training_only=True, resolution=100):
    df = check_data(df, prepared=True)
    # 使用joblib库进行并行计算
    n_cores = -1  # 使用所有可用CPU核心
    if variable is None:
        variable = feature_names

    results = Parallel(n_jobs=n_cores)(delayed(partial_dependencies_worker)(automl,df,feature_names,var) for var in variable)
    df_predict = pd.concat(results)
    df_predict.reset_index(drop=True, inplace=True)
    return df_predict


def partial_dependencies_worker(automl, df, model_features,variable,  training_only=True,
                                    resolution=100, n_cores=-1):
    # Filter only to training set
    if training_only:
        df = df[df["set"] == "training"]

    # Predict
    df_predict = pdp.pdp_isolate(model=automl, dataset=df,
                                 model_features=model_features,
                                 feature=variable, num_grid_points=10,grid_type='percentile',
                                 n_jobs=n_cores)

    # Alter names and add variable
    df_predict = pd.DataFrame({"value": df_predict.feature_grids,
                                "partial_dependency": df_predict.pdp})
    df_predict["variable"] = variable
    df_predict = df_predict[["variable", "value", "partial_dependency"]]

    # Catch factors, usually weekday
    if df_predict["value"].dtype == "object":
        df_predict["value"] = pd.to_numeric(df_predict["value"])

    return df_predict

def check_data(df, prepared):

    if 'date' not in df.columns:
        raise ValueError("Input must contain a `date` variable.")
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        raise ValueError("`date` variable needs to be a parsed date (datetime64).")
    if df['date'].isnull().any():
        raise ValueError("`date` must not contain missing (NA) values.")

    if prepared:
        if 'set' not in df.columns:
            raise ValueError("Input must contain a `set` variable.")
        if not set(df['set'].unique()).issubset(set(['training', 'testing'])):
            raise ValueError("`set` can only take the values `training` and `testing`.")
        if "value" not in df.columns:
            raise ValueError("Input must contain a `value` variable.")
        if "date_unix" not in df.columns:
            raise ValueError("Input must contain a `date_unix` variable.")
    return df
