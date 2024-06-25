normet.autodew.
==========================

.. function:: prepare_data(df, value, feature_names, na_rm=True, split_method='random', replace=False, fraction=0.75, seed=7654321)

    Prepares the input DataFrame by performing data cleaning, imputation, and splitting.

    :param df: Input DataFrame containing the dataset.
    :type df: pd.DataFrame
    :param value: Name of the target variable. Default is 'value'.
    :type value: str, optional
    :param feature_names: List of feature names. Default is None.
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
    :return: Prepared DataFrame with cleaned data and split into training and testing sets.
    :rtype: pd.DataFrame

    **Details:**

    - **Data Cleaning:** Checks the input data for consistency and validity using the `check_data` function.
    - **Imputation:** Handles missing values according to the `na_rm` parameter using the `impute_values` function.
    - **Date Variables:** Adds or replaces date-related variables in the dataset using the `add_date_variables` function.
    - **Data Splitting:** Splits the data into training and testing sets using the `split_into_sets` function based on the specified `split_method`.

    **Example Usage:**

    .. code-block:: python

        import pandas as pd
        from your_module import prepare_data

        df = pd.read_csv('timeseries_data.csv')
        value = 'target'
        feature_names = ['feature1', 'feature2', 'feature3']
        prepared_df = prepare_data(df, value, feature_names, split_method='time_series', fraction=0.8)

    **Notes:**

    - This function is a pipeline that sequentially applies various data preparation steps to ensure the dataset is clean and ready for modeling.
    - The `split_method` parameter allows flexibility in how the data is split, supporting both random and time-series based methods.
    - The `seed` parameter ensures reproducibility in random operations, particularly useful when `split_method` is 'random'.


.. function:: process_df(df, variables_col)

    Processes the DataFrame to ensure it contains necessary date and selected feature columns.

    This function checks if the date is present in the index or columns, selects the necessary features and
    the date column, and prepares the DataFrame for further analysis.

    :param df: Input DataFrame.
    :type df: pd.DataFrame
    :param variables_col: List of variable names to be included in the DataFrame.
    :type variables_col: list of str
    :returns: Processed DataFrame containing the date and selected feature columns.
    :rtype: pd.DataFrame
    :raises ValueError: If no datetime information is found in index or 'date' column.

    **Example:**

    .. code-block:: python

        df = pd.read_csv('data.csv')
        variables_col = ['feature1', 'feature2', 'feature3']
        processed_df = process_df(df, variables_col)


.. function:: check_data(df, value, feature_names)

    Validates and preprocesses the input DataFrame for subsequent analysis or modeling.

    :param df: Input DataFrame containing the data to be checked.
    :type df: pd.DataFrame
    :param value: Name of the target variable (column) to be used in the analysis.
    :type value: str
    :param feature_names: List of feature names to be included in the analysis. If empty, all columns are used.
    :type feature_names: list of str
    :returns: DataFrame containing only the necessary columns, with appropriate checks and transformations applied.
    :rtype: pd.DataFrame
    :raises ValueError: If any of the following conditions are met:
                       - The target variable (`value`) is not in the DataFrame columns.
                       - There is no datetime information in either the index or the 'date' column.
                       - The 'date' column is not of type datetime64.
                       - The 'date' column contains missing values.

    **Notes:**

    - If the DataFrame's index is a DatetimeIndex, it is reset to a column named 'date'.
    - The target column (`value`) is renamed to 'value'.
    - If `feature_names` is provided, only those columns (along with 'date' and the target column) are selected.

    **Example Usage:**

    .. code-block:: python

        df = pd.read_csv('data.csv')
        validated_df = check_data(df, value='target_variable', feature_names=['feature1', 'feature2'])

        print(validated_df.head())


.. function:: impute_values(df, na_rm)

    Imputes missing values in the DataFrame.

    :param df: Input DataFrame containing the dataset.
    :type df: pd.DataFrame
    :param na_rm: Whether to remove missing values.
    :type na_rm: bool
    :returns: DataFrame with imputed missing values.
    :rtype: pd.DataFrame

    **Details:**

    - Missing Values Handling: Depending on the value of `na_rm`, missing values can either be removed (`na_rm=True`) or imputed.
    - Numeric Variables: Missing values in numeric columns are filled with the median of each column.
    - Categorical Variables: Missing values in categorical columns (object or category dtype) are filled with the mode (most frequent value) of each column.

    **Example Usage:**

    .. code-block:: python

        df = pd.read_csv('data.csv')
        cleaned_df = impute_values(df, na_rm=True)

        print(cleaned_df.head())


.. function:: add_date_variables(df, replace)

    Adds date-related variables to the DataFrame.

    :param df: Input DataFrame containing the dataset.
    :type df: pd.DataFrame
    :param replace: Whether to replace existing date variables.
    :type replace: bool
    :returns: DataFrame with added date-related variables.
    :rtype: pd.DataFrame

    **Details:**

    - Date Variables Addition: Depending on the `replace` parameter, new date-related variables such as 'date_unix', 'day_julian', 'weekday', and 'hour' are added to the DataFrame.
    - Replace Existing Variables: If `replace=True`, existing date-related variables are overwritten with new values.
    - Non-replacement Logic: If `replace=False`, new date-related variables are added only if they do not already exist in the DataFrame.

    **Example Usage:**

    .. code-block:: python

        df = pd.read_csv('data.csv')
        enriched_df = add_date_variables(df, replace=True)

        print(enriched_df.head())


.. function:: split_into_sets(df, split_method, fraction, seed)

    Splits the DataFrame into training and testing sets.

    :param df: Input DataFrame containing the dataset.
    :type df: pd.DataFrame
    :param split_method: Method for splitting data ('random' or 'time_series').
    :type split_method: str
    :param fraction: Fraction of the dataset to be used for training.
    :type fraction: float
    :param seed: Seed for random operations.
    :type seed: int
    :returns: DataFrame with a 'set' column indicating the training or testing set.
    :rtype: pd.DataFrame

    **Details:**

    - Random Split: If `split_method` is 'random', the DataFrame is split randomly into training and testing sets based on the `fraction` parameter.
    - Time Series Split: If `split_method` is 'time_series', the DataFrame is split sequentially where the first `fraction` proportion is used for training.
    - 'set' Column Addition: A new column 'set' is added to indicate whether each row belongs to the 'training' or 'testing' set.
    - Sorting: The resulting DataFrame is sorted by 'date' to maintain chronological order after splitting.

    **Example Usage:**

    .. code-block:: python

        df = pd.read_csv('data.csv')
        df_split = split_into_sets(df, split_method='random', fraction=0.8, seed=42)

        print(df_split.head())

.. function:: train_model(df, value, variables, model_config=None, seed=7654321)

    Trains a machine learning model using the provided dataset and parameters.

    :param df: Input DataFrame containing the dataset.
    :type df: pd.DataFrame
    :param value: Name of the target variable.
    :type value: str
    :param variables: List of feature variables.
    :type variables: list of str
    :param model_config: Configuration dictionary for model training parameters, optional.
    :type model_config: dict, optional
    :param seed: Random seed for reproducibility, optional. Default is 7654321.
    :type seed: int, optional
    :returns: Trained ML model object.
    :rtype: object
    :raises ValueError: If `variables` contains duplicates or if any `variables` are not present in the DataFrame.

    **Details:**

    - Duplicate Check: Raises a ValueError if `variables` contain duplicate elements.
    - DataFrame Validation: Raises a ValueError if any `variables` are not present in the DataFrame columns.
    - Data Selection: If a 'set' column exists in the DataFrame, selects data labeled as 'training' for model training.
    - Default Model Configuration: Includes a default configuration for model training, specifying parameters such as time budget, metric, estimator list, task type, and verbosity.
    - Custom Model Configuration: Allows overriding default configuration using the `model_config` parameter.
    - AutoML Training: Initializes and trains an AutoML model using the selected configuration and provided seed.
    - Output: Prints the timestamp when training starts, the best model selected, and its corresponding parameters.

    **Example Usage:**

    .. code-block:: python

        df = pd.read_csv('data.csv')
        model = train_model(df, 'target', ['feature1', 'feature2'])

        # Using custom model configuration
        custom_config = {
            'time_budget': 120,
            'metric': 'rmse',
            'estimator_list': ['lgbm', 'rf'],
            'task': 'regression',
            'verbose': True
        }
        model_custom = train_model(df, 'target', ['feature1', 'feature2'], model_config=custom_config)

    **Notes:**

    - This function assumes the use of an AutoML framework for model training.
    - Adjustments to the default model configuration can be made by passing a dictionary through `model_config`.


.. function:: prepare_train_model(df, value, feature_names, split_method, fraction, model_config, seed)

    Prepares the data and trains a machine learning model using the specified configuration.

    This function combines data preparation and model training steps. It prepares the input DataFrame
    for training by selecting relevant columns and splitting the data, then trains a machine learning
    model using the provided configuration.

    :param df: The input DataFrame containing the data to be used for training.
    :type df: pandas.DataFrame
    :param value: The name of the target variable (column) to be predicted.
    :type value: str
    :param feature_names: A list of feature column names to be used in the training.
    :type feature_names: list of str
    :param split_method: The method to split the data ('random' or other supported methods).
    :type split_method: str
    :param fraction: The fraction of data to be used for training.
    :type fraction: float
    :param model_config: The configuration dictionary for the AutoML model training.
    :type model_config: dict
    :param seed: The random seed for reproducibility.
    :type seed: int
    :returns: The prepared DataFrame ready for model training and the trained machine learning model.
    :rtype: tuple (pandas.DataFrame, object)

    **Example:**

    .. code-block:: python

        df = pd.read_csv('timeseries_data.csv')
        value = 'target'
        feature_names = ['feature1', 'feature2', 'feature3']
        split_method = 'random'
        fraction = 0.75
        model_config = {...}
        seed = 7654321
        df_prepared, model = prepare_train_model(df, value, feature_names, split_method, fraction, model_config, seed)


.. function:: normalise_worker(index, df, model, variables_resample, replace, seed, verbose, weather_df=None)

    Worker function for parallel normalization of data using randomly resampled meteorological parameters
    from another weather DataFrame within its date range. If no weather DataFrame is provided,
    it defaults to using the input DataFrame.

    :param index: Index of the worker.
    :type index: int
    :param df: Input DataFrame containing the dataset.
    :type df: pd.DataFrame
    :param model: Trained ML model.
    :type model: object
    :param variables_resample: List of resampling variables.
    :type variables_resample: list of str
    :param replace: Whether to sample with replacement.
    :type replace: bool
    :param seed: Random seed.
    :type seed: int
    :param verbose: Whether to print progress messages.
    :type verbose: bool
    :param weather_df: Weather DataFrame containing the meteorological parameters, defaults to None.
    :type weather_df: pd.DataFrame, optional
    :returns: DataFrame containing normalized predictions.
    :rtype: pd.DataFrame

    **Details:**

    - Prints progress messages every fifth prediction if `verbose` is True.
    - Uses the `weather_df` to sample meteorological parameters, or the input `df` if `weather_df` is not provided.
    - Randomly samples observations within the weather DataFrame using the specified `seed` and `replace` parameters.
    - Applies the sampled meteorological parameters to the `df`.
    - Uses the provided `model` to make predictions on the adjusted `df`.
    - Constructs a DataFrame containing the dates, observed values, normalized predictions, and seed information.

    **Example Usage:**

    .. code-block:: python

        predictions = normalise_worker(
            index=1,
            df=my_dataframe,
            model=my_model,
            variables_resample=['temperature', 'humidity'],
            replace=False,
            seed=42,
            verbose=True,
            weather_df=my_weather_dataframe
        )

    **Notes:**

    - Useful for parallel processing tasks where normalization of predictions is required.
    - Ensures reproducibility by using the specified random `seed`.
    - Facilitates monitoring of progress during the normalization process.


.. function:: normalise(df, model, feature_names, variables_resample=None, n_samples=300, replace=True,
              aggregate=True, seed=7654321, n_cores=None, verbose=True, weather_df=None)

    Normalizes the dataset using the trained model.

    :param df: Input DataFrame containing the dataset.
    :type df: pd.DataFrame
    :param model: Trained ML model.
    :type model: object
    :param feature_names: List of feature names.
    :type feature_names: list of str
    :param variables_resample: List of resampling variables. If None, all feature_names except 'date_unix' are used.
    :type variables_resample: list of str, optional
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
    :param verbose: Whether to print progress messages. Default is True.
    :type verbose: bool, optional
    :param weather_df: DataFrame containing weather data for resampling. If None, `df` is used.
    :type weather_df: pd.DataFrame, optional
    :returns: DataFrame containing normalized predictions.
    :rtype: pd.DataFrame

    **Example:**

    .. code-block:: python

        df = pd.read_csv('timeseries_data.csv')
        model = train_model(df, 'target', feature_names)
        feature_names = ['feature1', 'feature2', 'feature3']
        variables_resample = ['feature1', 'feature2']
        normalized_df = normalise(df, model, feature_names, variables_resample)

    **Details:**

    - Uses `variables_resample` to resample meteorological parameters, or defaults to using `df` if `weather_df` is not provided.
    - Randomly samples observations within the `weather_df` using the specified `seed` and `replace` parameters.
    - Applies the sampled meteorological parameters to the `df`.
    - Uses the provided `model` to make predictions on the adjusted `df`.
    - Constructs a DataFrame containing the dates, observed values, normalized predictions, and seed information.
    - Aggregates the results if `aggregate` is True, otherwise returns detailed predictions.

    **Notes:**

    - Useful for normalizing predictions using parallel processing.
    - Ensures reproducibility by using the specified random `seed`.
    - Facilitates monitoring of progress during the normalization process.


.. function:: do_all(df=None, model=None, value=None, feature_names=None, variables_resample=None, split_method='random', fraction=0.75,
                     model_config=None, n_samples=300, seed=7654321, n_cores=-1, aggregate=True, weather_df=None)

    Conducts data preparation, model training, and normalization, returning the transformed dataset and model statistics.

    This function performs the entire pipeline from data preparation to model training and normalization using
    specified parameters and returns the transformed dataset along with model statistics.

    :param df: Input DataFrame containing the dataset.
    :type df: pd.DataFrame
    :param model: Pre-trained model to use for decomposition. If None, a new model will be trained. Default is None.
    :type model: object, optional
    :param value: Name of the target variable.
    :type value: str
    :param feature_names: List of feature names.
    :type feature_names: list of str
    :param variables_resample: List of variables for normalization.
    :type variables_resample: list of str
    :param split_method: Method for splitting data ('random' or 'time_series'). Default is 'random'.
    :type split_method: str, optional
    :param fraction: Fraction of the dataset to be used for training. Default is 0.75.
    :type fraction: float, optional
    :param model_config: Configuration dictionary for model training parameters.
    :type model_config: dict, optional
    :param n_samples: Number of samples for normalization. Default is 300.
    :type n_samples: int, optional
    :param seed: Seed for random operations. Default is 7654321.
    :type seed: int, optional
    :param n_cores: Number of CPU cores to be used for normalization (-1 for all available cores). Default is -1.
    :type n_cores: int, optional
    :param weather_df: DataFrame containing weather data for resampling. Default is None.
    :type weather_df: pd.DataFrame, optional
    :param aggregate: Whether to aggregate results. Default is True.
    :type aggregate: bool, optional
    :returns: Transformed dataset with normalized values and model statistics.
    :rtype: tuple(pd.DataFrame, pd.DataFrame)

    **Example:**

    .. code-block:: python

        df = pd.read_csv('timeseries_data.csv')
        value = 'target'
        feature_names = ['feature1', 'feature2', 'feature3']
        variables_resample = ['feature1', 'feature2']
        df_dew, mod_stats = do_all(df, value=value, feature_names=feature_names, variables_resample=variables_resample)

    **Notes:**

    - Uses specified parameters to prepare data, train the model, and normalize the dataset.
    - Ensures reproducibility by using the specified random `seed`.
    - Facilitates monitoring of progress during the normalization process if `verbose` is True.
    - If no pre-trained `model` is provided, a new model will be trained using the provided configuration.


.. function:: do_all_unc(df=None, value=None, feature_names=None, variables_resample=None, split_method='random', fraction=0.75,
                         model_config=None, n_samples=300, n_models=10, confidence_level=0.95, seed=7654321, n_cores=-1, weather_df=None)

    Performs uncertainty quantification by training multiple models with different random seeds and calculates statistical metrics.

    This function performs the entire pipeline from data preparation to model training and normalization, with an added step
    to quantify uncertainty by training multiple models using different random seeds. It returns a dataframe containing observed
    values, mean, standard deviation, median, confidence bounds, and weighted values, as well as a dataframe with model statistics.

    :param df: Input dataframe containing the time series data.
    :type df: pd.DataFrame
    :param value: Column name of the target variable.
    :type value: str
    :param feature_names: List of feature column names.
    :type feature_names: list of str
    :param variables_resample: List of sampled feature names for normalization.
    :type variables_resample: list of str
    :param split_method: Method to split the data ('random' or other methods). Default is 'random'.
    :type split_method: str, optional
    :param fraction: Fraction of data to be used for training. Default is 0.75.
    :type fraction: float, optional
    :param model_config: Configuration dictionary for model training parameters.
    :type model_config: dict, optional
    :param n_samples: Number of samples for normalization. Default is 300.
    :type n_samples: int, optional
    :param n_models: Number of models to train for uncertainty quantification. Default is 10.
    :type n_models: int, optional
    :param confidence_level: Confidence level for the uncertainty bounds. Default is 0.95.
    :type confidence_level: float, optional
    :param seed: Random seed for reproducibility. Default is 7654321.
    :type seed: int, optional
    :param n_cores: Number of cores to be used (-1 for all available cores). Default is -1.
    :type n_cores: int, optional
    :param weather_df: DataFrame containing weather data for resampling. Default is None.
    :type weather_df: pd.DataFrame, optional
    :returns: Dataframe with observed values, mean, standard deviation, median, lower and upper bounds, and weighted values, and model statistics.
    :rtype: tuple(pd.DataFrame, pd.DataFrame)

    **Example:**

    .. code-block:: python

        df = pd.read_csv('timeseries_data.csv')
        value = 'target'
        feature_names = ['feature1', 'feature2', 'feature3']
        variables_resample = ['feature1', 'feature2']
        df_dew, mod_stats = do_all_unc(df, value=value, feature_names=feature_names, variables_resample=variables_resample)

    **Notes:**

    - Uses specified parameters to prepare data, train the model, and normalize the dataset with uncertainty quantification.
    - Ensures reproducibility by using the specified random `seed`.
    - Facilitates monitoring of progress during the normalization process.
    - Trains multiple models with different random seeds to quantify uncertainty.
    - Calculates statistical metrics including mean, standard deviation, median, confidence bounds, and weighted values.


.. function:: decom_emi(df=None, model=None, value=None, feature_names=None, split_method='random', fraction=0.75, model_config=None, n_samples=300, seed=7654321, n_cores=-1)

    Decomposes a time series into different components using machine learning models.

    This function prepares the data, trains a machine learning model using AutoML, and decomposes the time series data into various components. The decomposition is based on the contribution of different features to the target variable. It returns the decomposed data and model statistics.

    :param df: Input dataframe containing the time series data.
    :type df: pd.DataFrame
    :param model: Pre-trained model to use for decomposition. If None, a new model will be trained. Default is None.
    :type model: object, optional
    :param value: Column name of the target variable.
    :type value: str
    :param feature_names: List of feature column names.
    :type feature_names: list of str
    :param split_method: Method to split the data ('random' or other methods). Default is 'random'.
    :type split_method: str, optional
    :param fraction: Fraction of data to be used for training. Default is 0.75.
    :type fraction: float, optional
    :param model_config: Configuration dictionary for model training parameters.
    :type model_config: dict, optional
    :param n_samples: Number of samples for normalization. Default is 300.
    :type n_samples: int, optional
    :param seed: Random seed for reproducibility. Default is 7654321.
    :type seed: int, optional
    :param n_cores: Number of cores to be used (-1 for all available cores). Default is -1.
    :type n_cores: int, optional
    :returns: A tuple containing a dataframe with decomposed components and a dataframe with model statistics.
    :rtype: tuple (pd.DataFrame, pd.DataFrame)

    **Example:**

    .. code-block:: python

        df = pd.read_csv('timeseries_data.csv')
        value = 'target'
        feature_names = ['feature1', 'feature2', 'feature3']
        df_dewc, mod_stats = decom_emi(df, value, feature_names)

    **Details:**

    - If no pre-trained model is provided, the function will prepare the data and train a new model using AutoML.
    - The function gathers model statistics for testing, training, and the entire dataset.
    - The time series is decomposed by excluding different features iteratively.
    - The decomposed components are adjusted to create deweathered values.
    - The results include the decomposed dataframe and model statistics for further analysis.


.. function:: decom_met(df=None, model=None, value=None, feature_names=None, split_method='random', fraction=0.75, model_config=None, n_samples=300, seed=7654321, importance_ascending=False, n_cores=-1)

    Decomposes a time series into different components using machine learning models with feature importance ranking.

    This function prepares the data, trains a machine learning model using AutoML, and decomposes the time series data into various components. The decomposition is based on the feature importance ranking and their contributions to the target variable. It returns the decomposed data and model statistics.

    :param df: Input dataframe containing the time series data.
    :type df: pd.DataFrame
    :param model: Pre-trained model to use for decomposition. If None, a new model will be trained. Default is None.
    :type model: object, optional
    :param value: Column name of the target variable.
    :type value: str
    :param feature_names: List of feature column names.
    :type feature_names: list of str
    :param split_method: Method to split the data ('random' or other methods). Default is 'random'.
    :type split_method: str, optional
    :param fraction: Fraction of data to be used for training. Default is 0.75.
    :type fraction: float, optional
    :param model_config: Configuration dictionary for model training parameters.
    :type model_config: dict, optional
    :param n_samples: Number of samples for normalization. Default is 300.
    :type n_samples: int, optional
    :param seed: Random seed for reproducibility. Default is 7654321.
    :type seed: int, optional
    :param importance_ascending: Sort order for feature importances. Default is False.
    :type importance_ascending: bool, optional
    :param n_cores: Number of cores to be used (-1 for all available cores). Default is -1.
    :type n_cores: int, optional
    :returns: A dataframe with decomposed components and a dataframe with model statistics.
    :rtype: tuple (pd.DataFrame, pd.DataFrame)

    **Example:**

    .. code-block:: python

        df = pd.read_csv('timeseries_data.csv')
        value = 'target'
        feature_names = ['feature1', 'feature2', 'feature3']
        df_dewwc, mod_stats = decom_met(df, value, feature_names)

    **Details:**

    - If no pre-trained model is provided, the function will prepare the data and train a new model using AutoML.
    - The function gathers model statistics for testing, training, and the entire dataset.
    - Feature importances are determined and sorted based on their contribution to the target variable.
    - The time series is decomposed by excluding different features iteratively, according to their importance.
    - The decomposed components are adjusted to create weather-independent values.
    - The results include the decomposed dataframe and model statistics for further analysis.


.. function:: rolling_dew(df=None, model=None, value=None, feature_names=None, variables_resample=None, split_method='random', fraction=0.75, model_config=None, n_samples=300, window_days=14, rollingevery=2, seed=7654321, n_cores=-1)

    Applies a rolling window approach to decompose the time series into different components using machine learning models.

    This function prepares the data, trains a machine learning model using AutoML, and applies a rolling window approach
    to decompose the time series data into various components. The decomposition is based on the contribution of different
    features to the target variable over rolling windows. It returns the decomposed data and model statistics.

    :param df: Input dataframe containing the time series data.
    :type df: pd.DataFrame
    :param model: Pre-trained model to use for decomposition. If None, a new model will be trained. Default is None.
    :type model: object, optional
    :param value: Column name of the target variable.
    :type value: str
    :param feature_names: List of feature column names.
    :type feature_names: list of str
    :param variables_resample: List of sampled feature names for normalization.
    :type variables_resample: list of str
    :param split_method: Method to split the data ('random' or other methods). Default is 'random'.
    :type split_method: str, optional
    :param fraction: Fraction of data to be used for training. Default is 0.75.
    :type fraction: float, optional
    :param model_config: Configuration dictionary for model training parameters.
    :type model_config: dict, optional
    :param n_samples: Number of samples for normalization. Default is 300.
    :type n_samples: int, optional
    :param window_days: Number of days for the rolling window. Default is 14.
    :type window_days: int, optional
    :param rollingevery: Rolling interval in days. Default is 2.
    :type rollingevery: int, optional
    :param seed: Random seed for reproducibility. Default is 7654321.
    :type seed: int, optional
    :param n_cores: Number of cores to be used (-1 for all available cores). Default is -1.
    :type n_cores: int, optional
    :returns: Tuple containing:
              - dfr (pd.DataFrame): Dataframe with rolling decomposed components.
              - mod_stats (pd.DataFrame): Dataframe with model statistics.

    **Details:**

    - Data Preparation: Prepares the input data for modeling and optionally trains a new model using AutoML.
    - Model Training: Trains or uses the provided model to learn the relationship between features and the target variable.
    - Rolling Window Decomposition: Applies a rolling window approach to decompose the time series into components over specified windows and intervals.
    - Feature Normalization: Normalizes the data within each rolling window using `normalise` function.
    - Returns decomposed data (`dfr`) and model statistics (`mod_stats`) for evaluation and analysis.

    **Example Usage:**

    - Useful for analyzing time series data with varying patterns over time and decomposing it into interpretable components.
    - Supports dynamic assessment of feature contributions to the target variable across different rolling windows.

    .. code-block:: python

        df = pd.read_csv('timeseries_data.csv')
        value = 'target'
        feature_names = ['feature1', 'feature2', 'feature3']
        variables_resample = ['feature1', 'feature2']
        dfr, mod_stats = rolling_dew(df, value, feature_names, variables_resample)

    **Notes:**

    - Enhances understanding of time series data by breaking down its components over sliding windows.
    - Facilitates evaluation of model performance and feature relevance across different temporal contexts.


.. function:: rolling_met(df=None, model=None, value=None, feature_names=None, split_method='random', fraction=0.75, model_config=None, n_samples=300, window_days=14, rollingevery=2, seed=7654321, n_cores=-1)

    Applies a rolling window approach to decompose the time series into different components using machine learning models.

    This function prepares the data, trains a machine learning model using AutoML, and applies a rolling window approach
    to decompose the time series data into various components. The decomposition is based on the contribution of different
    features to the target variable. It returns the decomposed data and model statistics.

    :param df: Input dataframe containing the time series data.
    :type df: pd.DataFrame
    :param model: Pre-trained model to use for decomposition. If None, a new model will be trained. Default is None.
    :type model: object, optional
    :param value: Column name of the target variable.
    :type value: str
    :param feature_names: List of feature column names.
    :type feature_names: list of str
    :param split_method: Method to split the data ('random' or other methods). Default is 'random'.
    :type split_method: str, optional
    :param fraction: Fraction of data to be used for training. Default is 0.75.
    :type fraction: float, optional
    :param model_config: Configuration dictionary for model training parameters.
    :type model_config: dict, optional
    :param n_samples: Number of samples for normalization. Default is 300.
    :type n_samples: int, optional
    :param window_days: Number of days for the rolling window. Default is 14.
    :type window_days: int, optional
    :param rollingevery: Rolling interval in days. Default is 2.
    :type rollingevery: int, optional
    :param seed: Random seed for reproducibility. Default is 7654321.
    :type seed: int, optional
    :param n_cores: Number of cores to be used (-1 for all available cores). Default is -1.
    :type n_cores: int, optional
    :returns: Tuple containing:
              - df_dew (pd.DataFrame): Dataframe with decomposed components including mean and standard deviation of the rolling window.
              - mod_stats (pd.DataFrame): Dataframe with model statistics.

    **Details:**

    - Data Preparation: Prepares the input data for modeling and optionally trains a new model using AutoML.
    - Model Training: Trains or uses the provided model to learn the relationship between features and the target variable.
    - Rolling Window Decomposition: Applies a rolling window approach to decompose the time series into components over specified windows and intervals.
    - Feature Normalization: Normalizes the data within each rolling window using `normalise` function.
    - Component Calculation: Calculates mean and standard deviation of the rolling window to derive short-term and seasonal components.
    - Returns decomposed data (`df_dew`) including observed, short-term, seasonal components, and statistics (`mod_stats`) for evaluation.

    **Example Usage:**

    - Useful for analyzing time series data with varying patterns over time and decomposing it into interpretable components.
    - Supports dynamic assessment of feature contributions to the target variable across different rolling windows.

    .. code-block:: python

        df = pd.read_csv('timeseries_data.csv')
        value = 'target'
        feature_names = ['feature1', 'feature2', 'feature3']
        df_dew, mod_stats = rolling_met(df, value, feature_names, window_days=14, rollingevery=2)

    **Notes:**

    - Enhances understanding of time series data by breaking down its components over sliding windows.
    - Facilitates evaluation of model performance and feature relevance across different temporal contexts.


.. function:: modStats(df, model, set=None, statistic=["n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA", "R2"])

    Calculates statistics for model evaluation based on provided data.

    :param df: Input DataFrame containing the dataset.
    :type df: pd.DataFrame
    :param model: Trained ML model.
    :type model: object
    :param set: Set type for which statistics are calculated ('training', 'testing', or 'all'). Default is None.
    :type set: str, optional
    :param statistic: List of statistics to calculate. Default is ["n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA", "R2"].
    :type statistic: list of str, optional
    :return: DataFrame containing calculated statistics.
    :rtype: pd.DataFrame

    **Example Usage:**

    Calculates statistics for a trained model on testing dataset:

    .. code-block:: python

        import pandas as pd
        from your_module import modStats, train_model

        df = pd.read_csv('timeseries_data.csv')
        model = train_model(df, 'target', feature_names)
        stats = modStats(df, model, set='testing')

    **Notes:**

    - If `set` parameter is provided, the function filters the DataFrame `df` to include only rows where the 'set' column matches `set`.
    - Raises a ValueError if `set` parameter is provided but 'set' column is not present in `df`.
    - Calculates statistics such as 'n', 'FAC2', 'MB', 'MGE', 'NMB', 'NMGE', 'RMSE', 'r', 'COE', 'IOA', 'R2' based on model predictions ('value_predict') and observed values ('value') in the DataFrame.


.. function:: Stats(df, mod, obs, statistic=["n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA", "R2"])

    Calculates specified statistics based on provided data.

    :param df: Input DataFrame containing the dataset.
    :type df: pd.DataFrame
    :param mod: Column name of the model predictions.
    :type mod: str
    :param obs: Column name of the observed values.
    :type obs: str
    :param statistic: List of statistics to calculate.
    :type statistic: list of str, optional
    :returns: DataFrame containing calculated statistics.
    :rtype: pd.DataFrame

    **Details:**

    This function calculates a range of statistical metrics to evaluate the model predictions against the observed values. The following statistics can be calculated:

    - **n**: Number of observations.
    - **FAC2**: Factor of 2.
    - **MB**: Mean Bias.
    - **MGE**: Mean Gross Error.
    - **NMB**: Normalized Mean Bias.
    - **NMGE**: Normalized Mean Gross Error.
    - **RMSE**: Root Mean Square Error.
    - **r**: Pearson correlation coefficient.
    - **COE**: Coefficient of Efficiency.
    - **IOA**: Index of Agreement.
    - **R2**: Coefficient of Determination (R-squared).

    The significance level of the correlation coefficient (p-value) is also evaluated and indicated with symbols:

    - `""` : p >= 0.1 (not significant)
    - `"+"` : 0.1 > p >= 0.05 (marginally significant)
    - `"*"` : 0.05 > p >= 0.01 (significant)
    - `"**"` : 0.01 > p >= 0.001 (highly significant)
    - `"***"` : p < 0.001 (very highly significant)

    **Example Usage:**

    .. code-block:: python

        import pandas as pd

        # Example DataFrame
        data = {
            'observed': [1, 2, 3, 4, 5],
            'predicted': [1.1, 1.9, 3.2, 3.8, 5.1]
        }
        df = pd.DataFrame(data)

        # Calculate statistics
        stats = Stats(df, mod='predicted', obs='observed')
        print(stats)

    **Notes:**

    - Each statistical metric has a specific function that calculates its value.
    - The function returns a DataFrame with the calculated statistics.
    - Significance levels for the correlation coefficient are marked with appropriate symbols.
