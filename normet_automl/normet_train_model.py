#require packages: import pandas as pd; from flaml import AutoML

def normet_train_model(df, variables,
    time_budget= 60,  # total running time in seconds
    metric= 'r2',  # primary metrics for regression can be chosen from: ['mae','mse','r2','rmse','mape']
    estimator_list= ["lgbm", "rf","xgboost","extra_tree","xgb_limitdepth"],  # list of ML learners; we tune lightgbm in this example
    task= 'regression',  # task type
    seed= 7654321,    # random seed
):
    # Check arguments
    if len(set(variables)) != len(variables):
        raise ValueError("`variables` contains duplicate elements.")

    if not all([var in df.columns for var in variables]):
        raise ValueError("`variables` given are not within input data frame.")

    # Check input dataset
    df = normet_check_data(df, prepared=True)

    # Filter and select input for modelling
    df = df.loc[df['set'] == 'training', ['value'] + variables]

    automl_settings = {
        "time_budget": time_budget,  # total running time in seconds
        "metric": metric,  # primary metrics for regression can be chosen from: ['mae','mse','r2','rmse','mape']
        "estimator_list": estimator_list,  # list of ML learners; we tune lightgbm in this example
        "task": task,  # task type
        "seed": seed,    # random seed
    }

    automl.fit(X_train=df[variables], y_train=df['value'],**automl_settings)

    return automl
