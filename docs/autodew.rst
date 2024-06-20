normet.autodew.
==========================

.. function:: emi_decom(df, value=None, feature_names=None, split_method='random', time_budget=60, metric='r2', estimator_list=["lgbm", "rf","xgboost","extra_tree","xgb_limitdepth"], task='regression', n_samples=300, fraction=0.75, seed=7654321, n_cores=-1)

    This function decomposes a time series into different components using machine learning models.

    :param df: Input dataframe containing the time series data.
    :type df: pd.DataFrame
    :param value: Column name of the target variable.
    :type value: str
    :param feature_names: List of feature column names.
    :type feature_names: list of str
    :param split_method: Method to split the data ('random' or other methods).
    :type split_method: str, optional
    :param time_budget: Time budget for the AutoML training.
    :type time_budget: int, optional
    :param metric: Metric to evaluate the model ('r2', 'mae', etc.).
    :type metric: str, optional
    :param estimator_list: List of estimators to be used in AutoML.
    :type estimator_list: list of str, optional
    :param task: Task type ('regression' or 'classification').
    :type task: str, optional
    :param n_samples: Number of samples for normalisation.
    :type n_samples: int, optional
    :param fraction: Fraction of data to be used for training.
    :type fraction: float, optional
    :param seed: Random seed for reproducibility.
    :type seed: int, optional
    :param n_cores: Number of cores to be used (-1 for all available cores).
    :type n_cores: int, optional
    :returns:
        - df_dewc: Dataframe with decomposed components.
        - mod_stats: Dataframe with model statistics.
    :rtype: tuple

    Example:

    .. code-block:: python

        df = pd.read_csv('timeseries_data.csv')
        value = 'target_variable'
        feature_names = ['feature1', 'feature2', 'feature3']
        df_dewc, mod_stats = ts_decom(df, value=value, feature_names=feature_names)

    This will decompose the time series data into different components using machine learning models, returning a dataframe with decomposed components and a dataframe with model statistics.

    **Details:**

    - **Data Preparation:** The input data is prepared using the `prepare_data` function.
    - **Model Training:** An AutoML model is trained using the specified features, time budget, and other parameters.
    - **Model Statistics:** Model statistics are generated for the testing, training, and all data sets.
    - **normalisation:** The data is normalized, and decomposed components are extracted.
    - **Decomposition:** The decomposed components are calculated, including adjustments for hour, weekday, and other factors.

    **Returns:**

    - `df_dewc`: A dataframe with decomposed components.
    - `mod_stats`: A dataframe with model statistics.

    **Example Usage:**

    .. code-block:: python

        df = pd.read_csv('timeseries_data.csv')
        df_dewc, mod_stats = ts_decom(df, value='target_variable', feature_names=['feature1', 'feature2'], split_method='random', time_budget=120, metric='mae', estimator_list=["lgbm", "xgboost"], task='regression', n_samples=500, fraction=0.8, seed=123456, n_cores=4)

        print(df_dewc.head())
        print(mod_stats.head())




.. function:: met_decom(df, value=None, feature_names=None, split_method='random', time_budget=60, metric='r2', estimator_list=["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"], task='regression', n_samples=300, fraction=0.75, seed=7654321, importance_ascending=False, n_cores=-1)

    Decomposes a time series into different components using machine learning models with feature importance ranking.

    :param df: Input dataframe containing the time series data.
    :type df: pd.DataFrame
    :param value: Column name of the target variable.
    :type value: str, optional
    :param feature_names: List of feature column names.
    :type feature_names: list of str, optional
    :param split_method: Method to split the data ('random' or other methods).
    :type split_method: str, optional
    :param time_budget: Time budget for the AutoML training.
    :type time_budget: int, optional
    :param metric: Metric to evaluate the model ('r2', 'mae', etc.).
    :type metric: str, optional
    :param estimator_list: List of estimators to be used in AutoML.
    :type estimator_list: list of str, optional
    :param task: Task type ('regression' or 'classification').
    :type task: str, optional
    :param n_samples: Number of samples for normalisation.
    :type n_samples: int, optional
    :param fraction: Fraction of data to be used for training.
    :type fraction: float, optional
    :param seed: Random seed for reproducibility.
    :type seed: int, optional
    :param importance_ascending: Sort order for feature importances.
    :type importance_ascending: bool, optional
    :param n_cores: Number of cores to be used (-1 for all available cores).
    :type n_cores: int, optional
    :returns:
        - df_dewwc: Dataframe with decomposed components.
        - mod_stats: Dataframe with model statistics.
    :rtype: tuple

    Applies a rolling window approach to decompose the time series into different components using machine learning models.

    :Example:

    .. code-block:: python

        df = pd.read_csv('timeseries_data.csv')
        value = 'target_variable'
        feature_names = ['feature1', 'feature2', 'feature3']
        df_dew, mod_stats = met_decom(df, value=value, feature_names=feature_names)

    This will apply a rolling window approach to decompose the time series data into different components using machine learning models, returning a dataframe with decomposed components and a dataframe with model statistics.

    **Details:**

    - **Data Preparation:** The input data is prepared using the `prepare_data` function.
    - **Model Training:** An AutoML model is trained using the specified features, time budget, and other parameters.
    - **Model Statistics:** Model statistics are generated for the testing, training, and all data sets.
    - **normalisation:** The data is normalized, and decomposed components are extracted.
    - **Rolling Window Decomposition:** The time series is decomposed using a rolling window approach, calculating the mean and standard deviation for each window.

    **Returns:**

    - `df_dewwc`: A dataframe with decomposed components.
    - `mod_stats`: A dataframe with model statistics.

    **Example Usage:**

    .. code-block:: python

        df = pd.read_csv('timeseries_data.csv')
        df_dewwc, mod_stats = met_decom(df, value='target_variable', feature_names=['feature1', 'feature2'], split_method='random', time_budget=120, metric='mae', estimator_list=["lgbm", "xgboost"], task='regression', n_samples=500, fraction=0.8, seed=123456, importance_ascending=True, n_cores=4)

        print(df_dewwc.head())
        print(mod_stats.head())



.. function:: rolling_dew(df, value=None, feature_names=None, split_method='random', time_budget=60, metric='r2', estimator_list=["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"], task='regression', variables_sample=None, n_samples=300, window_days=15, rollingevery=2, fraction=0.75, seed=7654321, n_cores=-1)

    Applies a rolling window approach to decompose the time series into different components using machine learning models.

    :param df: Input dataframe containing the time series data.
    :type df: pd.DataFrame
    :param value: Column name of the target variable.
    :type value: str, optional
    :param feature_names: List of feature column names.
    :type feature_names: list of str, optional
    :param split_method: Method to split the data ('random' or other methods).
    :type split_method: str, optional
    :param time_budget: Time budget for the AutoML training.
    :type time_budget: int, optional
    :param metric: Metric to evaluate the model ('r2', 'mae', etc.).
    :type metric: str, optional
    :param estimator_list: List of estimators to be used in AutoML.
    :type estimator_list: list of str, optional
    :param task: Task type ('regression' or 'classification').
    :type task: str, optional
    :param variables_sample: List of sampled feature names for normalisation (optional).
    :type variables_sample: list of str, optional
    :param n_samples: Number of samples for normalisation.
    :type n_samples: int, optional
    :param window_days: Number of days for the rolling window.
    :type window_days: int, optional
    :param rollingevery: Rolling interval.
    :type rollingevery: int, optional
    :param fraction: Fraction of data to be used for training.
    :type fraction: float, optional
    :param seed: Random seed for reproducibility.
    :type seed: int, optional
    :param n_cores: Number of cores to be used (-1 for all available cores).
    :type n_cores: int, optional
    :returns:
        - dfr: Dataframe with rolling decomposed components.
        - mod_stats: Dataframe with model statistics.
    :rtype: tuple

    **Details:**

    - **Data Preparation:** The input data is prepared using the `prepare_data` function.
    - **Model Training:** An AutoML model is trained using the specified features, time budget, and other parameters.
    - **Model Statistics:** Model statistics are generated for the testing, training, and all data sets.
    - **Rolling Window Decomposition:** The time series is decomposed using a rolling window approach, calculating the mean and standard deviation for each window.

    **Returns:**

    - `dfr`: A dataframe with rolling decomposed components.
    - `mod_stats`: A dataframe with model statistics.

    **Example Usage:**

    .. code-block:: python

        df = pd.read_csv('timeseries_data.csv')
        dfr, mod_stats = rolling_dew(df, value='target_variable', feature_names=['feature1', 'feature2'], split_method='random', time_budget=120, metric='mae', estimator_list=["lgbm", "xgboost"], task='regression', variables_sample=['feature1'], n_samples=500, window_days=30, rollingevery=5, fraction=0.8, seed=123456, n_cores=4)

        print(dfr.head())
        print(mod_stats.head())



.. function:: do_all_unc(df, value=None, feature_names=None, split_method='random', time_budget=60, metric='r2', estimator_list=["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"], task='regression', n_models=10, confidence_level=0.95, variables_sample=None, n_samples=300, fraction=0.75, seed=7654321, n_cores=-1)

    Performs uncertainty quantification by training multiple models with different random seeds and calculates statistical metrics.

    :param df: Input dataframe containing the time series data.
    :type df: pd.DataFrame
    :param value: Column name of the target variable.
    :type value: str, optional
    :param feature_names: List of feature column names.
    :type feature_names: list of str, optional
    :param split_method: Method to split the data ('random' or other methods).
    :type split_method: str, optional
    :param time_budget: Time budget for the AutoML training.
    :type time_budget: int, optional
    :param metric: Metric to evaluate the model ('r2', 'mae', etc.).
    :type metric: str, optional
    :param estimator_list: List of estimators to be used in AutoML.
    :type estimator_list: list of str, optional
    :param task: Task type ('regression' or 'classification').
    :type task: str, optional
    :param n_models: Number of models to train for uncertainty quantification.
    :type n_models: int, optional
    :param confidence_level: Confidence level for the uncertainty bounds.
    :type confidence_level: float, optional
    :param variables_sample: List of sampled feature names for normalisation (optional).
    :type variables_sample: list of str, optional
    :param n_samples: Number of samples for normalisation.
    :type n_samples: int, optional
    :param fraction: Fraction of data to be used for training.
    :type fraction: float, optional
    :param seed: Random seed for reproducibility.
    :type seed: int, optional
    :param n_cores: Number of cores to be used (-1 for all available cores).
    :type n_cores: int, optional
    :returns:
        - df_dew: Dataframe with observed values, mean, standard deviation, median, lower and upper bounds, and weighted values.
        - mod_stats: Dataframe with model statistics.
    :rtype: tuple

    **Details:**

    - **Random Seeds:** Random seeds are generated to train multiple models with different seeds for uncertainty quantification.
    - **Model Training:** Multiple models are trained using different random seeds and specified parameters.
    - **Statistical Metrics:** Statistical metrics are calculated for the observed values, mean, standard deviation, median, lower and upper bounds, and weighted values.
    - **Weighted Values:** Weighted values are calculated based on the R2 scores of the models.

    **Returns:**

    - `df_dew`: A dataframe with observed values, mean, standard deviation, median, lower and upper bounds, and weighted values.
    - `mod_stats`: A dataframe with model statistics.

    **Example Usage:**

    .. code-block:: python

        df = pd.read_csv('timeseries_data.csv')
        df_dew, mod_stats = do_all_unc(df, value='target_variable', feature_names=['feature1', 'feature2'], split_method='random', time_budget=120, metric='mae', estimator_list=["lgbm", "xgboost"], task='regression', n_models=5, confidence_level=0.90, variables_sample=['feature1'], n_samples=500, fraction=0.8, seed=123456, n_cores=4)

        print(df_dew.head())
        print(mod_stats.head())


.. function:: do_all(df, value=None, feature_names=None, split_method='random', time_budget=60, metric='r2', estimator_list=["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"], task='regression', variables_sample=None, n_samples=300, fraction=0.75, seed=7654321, n_cores=-1)

    Conducts data preparation, model training, and normalisation, returning the transformed dataset and model statistics.

    :param df: Input DataFrame containing the dataset.
    :type df: DataFrame
    :param value: Name of the target variable.
    :type value: str, optional
    :param feature_names: List of feature names.
    :type feature_names: list, optional
    :param split_method: Method for splitting data ('random' or 'time_series').
    :type split_method: str, optional
    :param time_budget: Maximum time allowed for training models, in seconds.
    :type time_budget: int, optional
    :param metric: Evaluation metric for model performance.
    :type metric: str, optional
    :param estimator_list: List of estimator names to be used in training.
    :type estimator_list: list, optional
    :param task: Task type ('regression' or 'classification').
    :type task: str, optional
    :param variables_sample: List of variables for normalisation.
    :type variables_sample: list, optional
    :param n_samples: Number of samples for normalisation.
    :type n_samples: int, optional
    :param fraction: Fraction of the dataset to be used for training.
    :type fraction: float, optional
    :param seed: Seed for random operations.
    :type seed: int, optional
    :param n_cores: Number of CPU cores to be used for normalisation.
    :type n_cores: int, optional
    :returns: Transformed dataset and model statistics DataFrame.
    :rtype: tuple

    **Details:**

    - **Data Preparation:** The input data is prepared using the `prepare_data` function.
    - **Model Training:** An AutoML model is trained using the specified features, time budget, and other parameters.
    - **Model Statistics:** Model statistics are generated for the testing, training, and all data sets.
    - **normalisation:** The data is normalized using the `normalise` function.

    **Returns:**

    - A tuple containing the transformed dataset and model statistics DataFrame.

    **Example Usage:**

    .. code-block:: python

        df = pd.read_csv('dataset.csv')
        df_dew, mod_stats = do_all(df, value='target', feature_names=['feat1', 'feat2'], split_method='random', time_budget=120, metric='r2', estimator_list=["lgbm", "rf"], task='regression', variables_sample=['feat1'], n_samples=500, fraction=0.8, seed=123456, n_cores=4)

        print(df_dew.head())
        print(mod_stats.head())



.. function:: prepare_data(df, value='value', feature_names=None, na_rm=True, split_method='random', replace=False, fraction=0.75, seed=7654321)

    Prepares the input DataFrame by performing data cleaning, imputation, and splitting.

    :param df: Input DataFrame containing the dataset.
    :type df: DataFrame
    :param value: Name of the target variable. Default is 'value'.
    :type value: str, optional
    :param feature_names: List of feature names.
    :type feature_names: list, optional
    :param na_rm: Whether to remove missing values. Default is True.
    :type na_rm: bool, optional
    :param split_method: Method for splitting data ('random' or 'time_series'). Default is 'random'.
    :type split_method: str, optional
    :param replace: Whether to replace existing date variables. Default is False.
    :type replace: bool, optional
    :param fraction: Fraction of the dataset to be used for training. Default is 0.75.
    :type fraction: float, optional
    :param seed: Seed for random operations. Default is 7654321.
    :type seed: int, optional
    :returns: Prepared DataFrame with cleaned data and split into training and testing sets.
    :rtype: DataFrame

    **Details:**

    - **Feature Selection:** Selects the relevant features from the DataFrame based on the provided feature names.
    - **Data Cleaning:** Performs data cleaning and imputation by removing or replacing missing values.
    - **Date Variables:** Adds date variables to the DataFrame, optionally replacing existing date variables.
    - **Data Splitting:** Splits the dataset into training and testing sets using the specified split method and fraction.

    **Returns:**

    - A DataFrame with cleaned data and split into training and testing sets.

    **Example Usage:**

    .. code-block:: python

        df = pd.read_csv('dataset.csv')
        prepared_df = prepare_data(df, value='target', feature_names=['feat1', 'feat2'], na_rm=True, split_method='random', replace=False, fraction=0.8, seed=123456)

        print(prepared_df.head())



.. function:: add_date_variables(df, replace)

    Adds date-related variables to the DataFrame.

    :param df: Input DataFrame containing the dataset.
    :type df: DataFrame
    :param replace: Whether to replace existing date variables.
    :type replace: bool
    :returns: DataFrame with added date-related variables.
    :rtype: DataFrame

    **Details:**

    - **Date Variables:** Adds date-related variables such as 'date_unix', 'day_julian', 'weekday', and 'hour' to the DataFrame.
    - **Replacement:** Determines whether to replace existing date variables if they already exist in the DataFrame.

    **Returns:**

    - A DataFrame with added date-related variables.

    **Example Usage:**

    .. code-block:: python

        df = pd.read_csv('dataset.csv')
        new_df = add_date_variables(df, replace=True)

        print(new_df.head())



.. function:: impute_values(df, na_rm)

    Imputes missing values in the DataFrame.

    :param df: Input DataFrame containing the dataset.
    :type df: DataFrame
    :param na_rm: Whether to remove missing values.
    :type na_rm: bool
    :returns: DataFrame with imputed missing values.
    :rtype: DataFrame

    **Details:**

    - **Missing Value Removal:** Removes missing values from the DataFrame if `na_rm` is set to True.
    - **Imputation:** Imputes missing numeric values with the median and missing categorical values with the mode.

    **Returns:**

    - A DataFrame with imputed missing values.

    **Example Usage:**

    .. code-block:: python

        df = pd.read_csv('dataset.csv')
        new_df = impute_values(df, na_rm=True)

        print(new_df.head())



.. function:: split_into_sets(df, split_method, fraction, seed)

    Splits the DataFrame into training and testing sets.

    :param df: Input DataFrame containing the dataset.
    :type df: DataFrame
    :param split_method: Method for splitting data ('random' or 'time_series').
    :type split_method: str
    :param fraction: Fraction of the dataset to be used for training.
    :type fraction: float
    :param seed: Seed for random operations.
    :type seed: int
    :returns: DataFrame with a 'set' column indicating the training or testing set.
    :rtype: DataFrame

    **Details:**

    - **Random Splitting:** If the split method is 'random', samples the dataset to create a training set of the specified fraction and assigns the rest to the testing set.
    - **Time Series Splitting:** If the split method is 'time_series', splits the dataset based on the fraction of rows specified, maintaining the temporal order of the data.

    **Returns:**

    - A DataFrame with a 'set' column indicating the training or testing set.

    **Example Usage:**

    .. code-block:: python

        df = pd.read_csv('dataset.csv')
        split_df = split_into_sets(df, split_method='random', fraction=0.8, seed=12345)

        print(split_df.head())



.. function:: check_data(df, prepared)

    Checks the integrity of the input DataFrame.

    :param df: Input DataFrame containing the dataset.
    :type df: DataFrame
    :param prepared: Whether the DataFrame is already prepared.
    :type prepared: bool
    :returns: DataFrame with checked integrity.
    :rtype: DataFrame

    **Details:**

    - **Date Variable Check:** Ensures that the input DataFrame contains a 'date' variable of type np.datetime64 without missing values.
    - **Prepared DataFrame Check:** If the DataFrame is marked as prepared, checks for additional required variables such as 'set', 'value', and 'date_unix'.

    **Returns:**

    - A DataFrame with checked integrity.

    **Example Usage:**

    .. code-block:: python

        df = pd.read_csv('dataset.csv')
        checked_df = check_data(df, prepared=True)

        print(checked_df.head())


.. function:: train_model(df, variables, time_budget=60, metric='r2', estimator_list=["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"], task='regression', seed=7654321, verbose=True)

    Trains a model using the provided dataset and arguments.

    :param df: Input DataFrame containing the dataset.
    :type df: DataFrame
    :param variables: List of feature variables.
    :type variables: list
    :param time_budget: Total running time in seconds. Default is 60.
    :type time_budget: int, optional
    :param metric: Primary metric for regression. Default is 'r2'.
    :type metric: str, optional
    :param estimator_list: List of ML learners. Default is ["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"].
    :type estimator_list: list, optional
    :param task: Task type. Default is 'regression'.
    :type task: str, optional
    :param seed: Random seed. Default is 7654321.
    :type seed: int, optional
    :param verbose: Whether to print progress messages. Default is True.
    :type verbose: bool, optional
    :returns: Trained model.
    :rtype: object

    **Details:**

    - **Argument Validation:** Validates input arguments such as ensuring no duplicate elements in `variables`.
    - **Input Dataset Check:** Verifies the integrity of the input dataset.
    - **Model Training:** Trains a model using the specified features and settings.

    **Returns:**

    - The trained model.

    **Example Usage:**

    .. code-block:: python

        df = pd.read_csv('dataset.csv')
        variables = ['feature1', 'feature2', 'feature3']
        trained_model = train_model(df, variables)

        print(trained_model)




.. function:: normalise(automl, df, feature_names, variables=None, n_samples=300, replace=True, aggregate=True, seed=7654321, n_cores=None, verbose=False)

    Normalizes the dataset using the trained model.

    :param automl: Trained AutoML model.
    :type automl: object
    :param df: Input DataFrame containing the dataset.
    :type df: DataFrame
    :param feature_names: List of feature names.
    :type feature_names: list
    :param variables: List of feature variables. Default is None.
    :type variables: list, optional
    :param n_samples: Number of samples to normalize. Default is 300.
    :type n_samples: int, optional
    :param replace: Whether to replace existing data. Default is True.
    :type replace: bool, optional
    :param aggregate: Whether to aggregate results. Default is True.
    :type aggregate: bool, optional
    :param seed: Random seed. Default is 7654321.
    :type seed: int, optional
    :param n_cores: Number of CPU cores to use. Default is None.
    :type n_cores: int, optional
    :param verbose: Whether to print progress messages. Default is False.
    :type verbose: bool, optional
    :returns: DataFrame containing normalized predictions.
    :rtype: DataFrame

    **Details:**

    - **Input DataFrame Check:** Verifies the integrity of the input DataFrame.
    - **Variables Selection:** Selects variables for normalisation, defaulting to all features except 'date_unix'.
    - **Sampling and Prediction:** Samples the time series and predicts normalisation using the trained model.
    - **Parallelization:** Utilizes parallel processing for faster execution based on the number of CPU cores.
    - **Result Aggregation:** Aggregates the results into a DataFrame containing normalized predictions.

    **Returns:**

    - A DataFrame containing normalized predictions.

    **Example Usage:**

    .. code-block:: python

        df = pd.read_csv('dataset.csv')
        feature_names = ['feature1', 'feature2', 'feature3']
        trained_automl = train_model(df, feature_names)
        normalized_df = normalise(trained_automl, df, feature_names)

        print(normalized_df.head())



.. function:: modStats(df, set=set, statistic=["n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA", "R2"])

    Calculates statistics for model evaluation based on provided data.

    :param df: Input DataFrame containing the dataset.
    :type df: DataFrame
    :param set: Set type for which statistics are calculated ('training', 'testing', or 'all').
    :type set: str
    :param statistic: List of statistics to calculate.
    :type statistic: list
    :returns: DataFrame containing calculated statistics.
    :rtype: DataFrame

    **Details:**

    - **Subset Selection:** Filters the DataFrame based on the provided set type.
    - **Model Prediction:** Uses the trained AutoML model to predict values.
    - **Statistical Calculation:** Calculates various statistics based on provided parameters.
    - **Assignment:** Assigns the set type to the resulting DataFrame.

    **Returns:**

    - A DataFrame containing calculated statistics.

    **Example Usage:**

    This function is typically used to evaluate model performance by calculating various statistics such as RMSE, R2, etc., based on the provided dataset and set type.
