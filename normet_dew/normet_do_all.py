def normet_do_all(df, value=None,feature_names=None, split_method = 'random',time_budget=60,metric= 'r2',
                  estimator_list=["lgbm", "rf","xgboost","extra_tree","xgb_limitdepth"],task='regression',
                  seed=7654321, variables_sample=None, n_samples=300,fraction=0.75):
    df=normet_prepare_data(df, value=value, split_method = split_method,fraction=fraction)
    automl=normet_train_model(df,variables=feature_names,
                time_budget= time_budget,  metric= metric, task= task, seed= seed);
    df=normet_normalise(automl, df,
                           feature_names = feature_names,
                          variables= variables_sample,
                          n_samples=n_samples)
    return df
