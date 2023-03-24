#from joblib import Parallel, delayed
#from pdpbox import pdp
def normet_partial_dependencies(automl, df, variable=None, training_only=True, resolution=100):
    df = normet_check_data(df, prepared=True)
    # 使用joblib库进行并行计算
    n_cores = -1  # 使用所有可用CPU核心
    if variable is None:
        variable = feature_names

    results = Parallel(n_jobs=n_cores)(delayed(normet_partial_dependencies_worker)(automl,df,var) for var in variable)
    df_predict = pd.concat(results)
    df_predict.reset_index(drop=True, inplace=True)
    return df_predict

def normet_partial_dependencies_worker(automl, df, variable, training_only=True,
                                    resolution=100, n_cores=-1):
    # Filter only to training set
    if training_only:
        df = df[df["set"] == "training"]

    # Predict
    df_predict = pdp.pdp_isolate(model=automl, dataset=df,
                                 model_features=feature_names,
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
