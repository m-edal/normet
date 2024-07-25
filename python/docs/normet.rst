normet
==========================

.. function:: prepare_data(df, value, feature_names, na_rm=True, split_method='random', replace=False, fraction=0.75, seed=7654321)

    Prepares the input DataFrame by performing data cleaning, imputation, and splitting.

    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
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
    :rtype: pandas.DataFrame

    **Details:**

    - **Data Cleaning:** Checks the input data for consistency and validity using the `check_data` function.
    - **Imputation:** Handles missing values according to the `na_rm` parameter using the `impute_values` function.
    - **Date Variables:** Adds or replaces date-related variables in the dataset using the `add_date_variables` function.
    - **Data Splitting:** Splits the data into training and testing sets using the `split_into_sets` function based on the specified `split_method`.

    **Example:**

    .. code-block:: python

        import pandas as pd
        import normet as nm
        df = pd.read_csv('timeseries_data.csv')
        value = 'target'
        feature_names = ['feature1', 'feature2', 'feature3']
        prepared_df = nm.prepare_data(df, value, feature_names, split_method='time_series', fraction=0.8)

    **Notes:**

    - This function is a pipeline that sequentially applies various data preparation steps to ensure the dataset is clean and ready for modeling.
    - The `split_method` parameter allows flexibility in how the data is split, supporting both random and time-series based methods.
    - The `seed` parameter ensures reproducibility in random operations, particularly useful when `split_method` is 'random'.


.. function:: process_df(df, variables_col)

    Processes the DataFrame to ensure it contains necessary date and selected feature columns.

    This function checks if the date is present in the index or columns, selects the necessary features and
    the date column, and prepares the DataFrame for further analysis.

    :param df: Input DataFrame.
    :type df: pandas.DataFrame
    :param variables_col: List of variable names to be included in the DataFrame.
    :type variables_col: list of str
    :returns: Processed DataFrame containing the date and selected feature columns.
    :rtype: pandas.DataFrame
    :raises ValueError: If no datetime information is found in index or 'date' column.

    **Example:**

    .. code-block:: python

        import pandas as pd
        import normet as nm
        df = pd.read_csv('data.csv')
        variables_col = ['feature1', 'feature2', 'feature3']
        processed_df = nm.process_df(df, variables_col)


.. function:: check_data(df, value)

    Validates and preprocesses the input DataFrame for subsequent analysis or modeling.

    This function checks if the target variable is present, ensures the date column is of the correct type, and validates there are no missing dates, returning a DataFrame with the target column renamed for consistency.

    :param df: Input DataFrame containing the data to be checked.
    :type df: pandas.DataFrame
    :param value: Name of the target variable (column) to be used in the analysis.
    :type value: str
    :returns: A DataFrame containing only the necessary columns, with appropriate checks and transformations applied.
    :rtype: pandas.DataFrame
    :raises ValueError:
        - If the target variable (`value`) is not in the DataFrame columns.
        - If there is no datetime information in either the index or the 'date' column.
        - If the 'date' column is not of type datetime64.
        - If the 'date' column contains missing values.

    :notes:
        - If the DataFrame's index is a DatetimeIndex, it is reset to a column named 'date'.
        - The target column (`value`) is renamed to 'value'.

    **Example:**

    .. code-block:: python

        import pandas as pd
        import normet as nm
        data = {
             'timestamp': pd.date_range(start='1/1/2020', periods=5, freq='D'),
             'target': [1, 2, 3, 4, 5]
         }
        df = pd.DataFrame(data).set_index('timestamp')
        df_checked = nm.check_data(df, 'target')
        print(df_checked)


.. function:: impute_values(df, na_rm)

    Imputes missing values in the DataFrame.

    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param na_rm: Whether to remove missing values.
    :type na_rm: bool
    :returns: DataFrame with imputed missing values.
    :rtype: pandas.DataFrame

    **Details:**

    - Missing Values Handling: Depending on the value of `na_rm`, missing values can either be removed (`na_rm=True`) or imputed.
    - Numeric Variables: Missing values in numeric columns are filled with the median of each column.
    - Categorical Variables: Missing values in categorical columns (object or category dtype) are filled with the mode (most frequent value) of each column.

    **Example:**

    .. code-block:: python

        import pandas as pd
        import normet as nm
        df = pd.read_csv('data.csv')
        cleaned_df = nm.impute_values(df, na_rm=True)
        print(cleaned_df.head())


.. function:: add_date_variables(df, replace)

    Adds date-related variables to the DataFrame.

    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param replace: Whether to replace existing date variables.
    :type replace: bool
    :returns: DataFrame with added date-related variables.
    :rtype: pandas.DataFrame

    **Details:**

    - Date Variables Addition: Depending on the `replace` parameter, new date-related variables such as 'date_unix', 'day_julian', 'weekday', and 'hour' are added to the DataFrame.
    - Replace Existing Variables: If `replace=True`, existing date-related variables are overwritten with new values.
    - Non-replacement Logic: If `replace=False`, new date-related variables are added only if they do not already exist in the DataFrame.

    **Example:**

    .. code-block:: python

        import pandas as pd
        import normet as nm
        df = pd.read_csv('data.csv')
        enriched_df = nm.add_date_variables(df, replace=True)
        print(enriched_df.head())


.. function:: split_into_sets(df, split_method, fraction, seed)

    Splits the DataFrame into training and testing sets based on the specified split method.

    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param split_method: Method for splitting data ('random', 'ts', 'season', 'month').
    :type split_method: str
    :param fraction: Fraction of the dataset to be used for training (for 'random', 'ts', 'season') or fraction of each month to be used for training (for 'month').
    :type fraction: float
    :param seed: Seed for random operations.
    :type seed: int

    :returns: DataFrame with a 'set' column indicating the training or testing set.
    :rtype: pandas.DataFrame

    **Example:**

    .. code-block:: python

        import pandas as pd
        import normet as nm
        data = {
             'date': pd.date_range(start='2020-01-01', periods=365),
             'value': range(365)
         }
        df = pd.DataFrame(data)
        df_split = nm.split_into_sets(df, split_method='season', fraction=0.8, seed=12345)

    **Notes:**

    - Depending on the `split_method`:
        - 'random': Randomly splits the data into training and testing sets.
        - 'ts': Splits the data based on a fraction of the total length.
        - 'season': Splits the data into seasonal sets based on the month of the year.
        - 'month': Splits the data into monthly sets.
    - Each resulting DataFrame will have a 'set' column indicating whether the row belongs to the 'training' or 'testing' set.


.. function:: train_model(df, value='value', variables=None, model_config=None, seed=7654321, verbose=True)

    Trains a machine learning model using the provided dataset and parameters.

    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param value: Name of the target variable. Default is 'value'.
    :type value: str, optional
    :param variables: List of feature variables. Default is None.
    :type variables: list of str
    :param model_config: Configuration dictionary for model training parameters.
    :type model_config: dict, optional
    :param seed: Random seed for reproducibility. Default is 7654321.
    :type seed: int, optional
    :param verbose: If True, print progress messages. Default is True.
    :type verbose: bool, optional

    :returns: Trained ML model object.
    :rtype: object
    :raises ValueError: If `variables` contains duplicates or if any `variables` are not present in the DataFrame.

    **Example:**

    .. code-block:: python

        import pandas as pd
        import normet as nm
        data = {
             'feature1': [1, 2, 3, 4, 5],
             'feature2': [5, 4, 3, 2, 1],
             'target': [10, 20, 30, 40, 50],
             'set': ['training', 'training', 'training', 'validation', 'validation']
         }
        df = pd.DataFrame(data)
        model = nm.train_model(df, value='target', variables=['feature1', 'feature2'])

    **Notes:**

    - If the 'set' column is present in the DataFrame, only rows where `set` is 'training' are used for training.
    - The default `model_config` includes:

    .. code-block:: python

        model_config = {
        'time_budget': 90,                     # Total running time in seconds
        'metric': 'r2',                        # Primary metric for regression, 'mae', 'mse', 'r2', 'mape',...
        'estimator_list': ["lgbm"],            # List of ML learners: ["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"]
        'task': 'regression',                  # Task type
        'verbose': verbose                     # Print progress messages
        }

    - This configuration can be updated with user-provided `model_config`.


.. function:: prepare_train_model(df, value, feature_names, split_method, fraction, model_config, seed, verbose=True)

    Prepares the data and trains a machine learning model using the specified configuration.

    :param df: The input DataFrame containing the data to be used for training.
    :type df: pandas.DataFrame
    :param value: The name of the target variable to be predicted.
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
    :param verbose: If True, print progress messages. Default is True.
    :type verbose: bool, optional

    :returns: A tuple containing:
        - pd.DataFrame: The prepared DataFrame ready for model training.
        - object: The trained machine learning model.
    :rtype: tuple

    :raises ValueError: If there are any issues with the data preparation or model training.

    **Example:**

    .. code-block:: python

        import pandas as pd
        import normet as nm
        data = {
             'feature1': [1, 2, 3, 4, 5],
             'feature2': [5, 4, 3, 2, 1],
             'target': [2, 3, 4, 5, 6],
             'set': ['training', 'training', 'training', 'testing', 'testing']
         }
        df = pd.DataFrame(data)
        feature_names = ['feature1', 'feature2']
        split_method = 'random'
        fraction = 0.75
        model_config = {'time_budget': 90, 'metric': 'r2'}
        seed = 7654321
        df_prepared, model = nm.prepare_train_model(df, value='target', feature_names=feature_names, split_method=split_method, fraction=fraction, model_config=model_config, seed=seed, verbose=True)

    **Notes:**

    - The `prepare_data` function is called to preprocess and split the data based on the given `split_method` and `fraction`.
    - The `train_model` function is then used to train the model using the prepared data and specified `model_config`.
    - The default `model_config` includes:

    .. code-block:: python

        model_config = {
        'time_budget': 90,                     # Total running time in seconds
        'metric': 'r2',                        # Primary metric for regression, 'mae', 'mse', 'r2', 'mape',...
        'estimator_list': ["lgbm"],            # List of ML learners: "lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"
        'task': 'regression',                  # Task type
        'verbose': verbose                     # Print progress messages
        }

    - The configuration for ML can be updated with user-provided `model_config`.
    - Any columns named 'date_unix', 'day_julian', 'weekday', or 'hour' are excluded from the feature variables before preparing the data.


.. function:: normalise_worker(index, df, model, variables_resample, replace, seed, verbose, weather_df=None)

    Worker function for parallel normalisation of data using randomly resampled meteorological parameters
    from another weather DataFrame within its date range. If no weather DataFrame is provided, it defaults to using the input DataFrame.

    :param index: Index of the worker.
    :type index: int
    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
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
    :param weather_df: Weather DataFrame containing the meteorological parameters. Defaults to None.
    :type weather_df: pandas.DataFrame, optional

    :returns: DataFrame containing normalised predictions.
    :rtype: pandas.DataFrame

    **Example:**

    .. code-block:: python

        import pandas as pd
        import normet as nm
        data = {
             'date': pd.date_range(start='2020-01-01', periods=365),
             'value': range(365),
             'temp': np.random.rand(365),
             'humidity': np.random.rand(365)
         }
        weather_data = {
             'temp': np.random.rand(100),
             'humidity': np.random.rand(100)
         }
        df = pd.DataFrame(data)
        weather_df = pd.DataFrame(weather_data)
        model = nm.trained_model  # Assuming a trained model is available
        predictions = nm.normalise_worker(
             index=0,
             df=df,
             model=model,
             variables_resample=['temp', 'humidity'],
             replace=True,
             seed=42,
             verbose=True,
             weather_df=weather_df
         )
        print(predictions)

    **Notes:**

    - Progress messages are printed every fifth prediction if `verbose` is set to True.
    - Meteorological parameters are resampled either from the provided `weather_df` or the input `df` if `weather_df` is not provided.
    - The function returns a DataFrame with the original date, observed values, normalised predictions, and the seed used for random sampling.


.. function:: normalise(df, model, feature_names, variables_resample=None, n_samples=300, replace=True, aggregate=True, seed=7654321, n_cores=None, weather_df=None, verbose=True)

    Normalises the dataset using a trained machine learning model and optionally resamples meteorological parameters from a provided weather DataFrame.

    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param model: Trained ML model.
    :type model: object
    :param feature_names: List of feature names.
    :type feature_names: list of str
    :param variables_resample: List of resampling variables. Default is None.
    :type variables_resample: list of str, optional
    :param n_samples: Number of samples to normalise. Default is 300.
    :type n_samples: int, optional
    :param replace: Whether to replace existing data. Default is True.
    :type replace: bool, optional
    :param aggregate: Whether to aggregate results. Default is True.
    :type aggregate: bool, optional
    :param seed: Random seed. Default is 7654321.
    :type seed: int, optional
    :param n_cores: Number of CPU cores to use. Default is total CPU cores minus one.
    :type n_cores: int, optional
    :param weather_df: DataFrame containing weather data for resampling. Default is None.
    :type weather_df: pandas.DataFrame, optional
    :param verbose: Whether to print progress messages. Default is True.
    :type verbose: bool, optional

    :returns: DataFrame containing normalised predictions.
    :rtype: pandas.DataFrame

    **Example:**

    .. code-block:: python

        import pandas as pd
        import normet as nm
        data = {
             'date': pd.date_range(start='2020-01-01', periods=5, freq='D'),
             'feature1': [1, 2, 3, 4, 5],
             'feature2': [5, 4, 3, 2, 1],
             'value': [2, 3, 4, 5, 6]
         }
        df = pd.DataFrame(data)
        feature_names = ['feature1', 'feature2']
        model = nm.train_model(df, value='value', variables=feature_names)
        variables_resample = ['feature1', 'feature2']
        normalised_df = nm.normalise(df, model, feature_names, variables_resample)

    **Notes:**

    - The function can optionally use a separate weather DataFrame for resampling meteorological parameters.
    - Progress messages are printed if `verbose` is set to True.
    - The number of CPU cores used for parallel processing can be specified, or defaults to the total number of cores minus one.
    - If `aggregate` is True, the results are averaged; otherwise, the function returns all individual predictions.


.. function:: do_all(df=None, model=None, value=None, feature_names=None, variables_resample=None, split_method='random', fraction=0.75, model_config=None, n_samples=300, seed=7654321, n_cores=None, aggregate=True, weather_df=None, verbose=True)

    Conducts data preparation, model training, and normalisation, returning the transformed dataset and model statistics.

    This function performs the entire pipeline from data preparation to model training and normalisation using specified parameters and returns the transformed dataset along with model statistics.

    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param model: Pre-trained model to use for decomposition. If None, a new model will be trained. Default is None.
    :type model: object, optional
    :param value: Name of the target variable.
    :type value: str
    :param feature_names: List of feature names.
    :type feature_names: list of str
    :param variables_resample: List of variables for normalisation.
    :type variables_resample: list of str
    :param split_method: Method for splitting data ('random' or 'time_series'). Default is 'random'.
    :type split_method: str, optional
    :param fraction: Fraction of the dataset to be used for training. Default is 0.75.
    :type fraction: float, optional
    :param model_config: Configuration dictionary for model training parameters.
    :type model_config: dict, optional
    :param n_samples: Number of samples for normalisation. Default is 300.
    :type n_samples: int, optional
    :param seed: Seed for random operations. Default is 7654321.
    :type seed: int, optional
    :param n_cores: Number of CPU cores to be used for normalisation. Default is total CPU cores minus one.
    :type n_cores: int, optional
    :param weather_df: DataFrame containing weather data for resampling. Default is None.
    :type weather_df: pandas.DataFrame, optional
    :param verbose: Whether to print progress messages. Default is True.
    :type verbose: bool, optional

    :returns: Transformed dataset with normalised values and DataFrame containing model statistics.
    :rtype: tuple (pandas.DataFrame, pandas.DataFrame)

    **Example:**

    .. code-block:: python

        import pandas as pd
        import normet as nm
        df = pd.read_csv('timeseries_data.csv')
        value = 'target'
        feature_names = ['feature1', 'feature2', 'feature3']
        variables_resample = ['feature1', 'feature2']
        df_dew, mod_stats = nm.do_all(df, value=value, feature_names=feature_names, variables_resample=variables_resample)

    **Notes:**

    - If a model is not provided, the function will train a new model using the specified parameters.
    - Model statistics are collected for testing, training, and the entire dataset.
    - The function uses the specified number of CPU cores for normalisation, defaulting to one less than the total number of cores.
    - If a weather DataFrame is provided, it is used for resampling meteorological parameters; otherwise, the input DataFrame is used.
    - Progress messages are printed if `verbose` is set to True.


.. function:: do_all_unc(df=None, value=None, feature_names=None, variables_resample=None, split_method='random', fraction=0.75, model_config=None, n_samples=300, n_models=10, confidence_level=0.95, seed=7654321, n_cores=None, weather_df=None, verbose=True)

    Performs uncertainty quantification by training multiple models with different random seeds and calculates statistical metrics.

    :param df: Input dataframe containing the time series data.
    :type df: pandas.DataFrame
    :param value: Column name of the target variable.
    :type value: str
    :param feature_names: List of feature column names.
    :type feature_names: list of str
    :param variables_resample: List of sampled feature names for normalisation.
    :type variables_resample: list of str
    :param split_method: Method to split the data ('random' or other methods). Default is 'random'.
    :type split_method: str, optional
    :param fraction: Fraction of data to be used for training. Default is 0.75.
    :type fraction: float, optional
    :param model_config: Configuration dictionary for model training parameters.
    :type model_config: dict, optional
    :param n_samples: Number of samples for normalisation. Default is 300.
    :type n_samples: int, optional
    :param n_models: Number of models to train for uncertainty quantification. Default is 10.
    :type n_models: int, optional
    :param confidence_level: Confidence level for the uncertainty bounds. Default is 0.95.
    :type confidence_level: float, optional
    :param seed: Random seed for reproducibility. Default is 7654321.
    :type seed: int, optional
    :param n_cores: Number of cores to be used. Default is total CPU cores minus one.
    :type n_cores: int, optional
    :param weather_df: DataFrame containing weather data for resampling. Default is None.
    :type weather_df: pandas.DataFrame, optional
    :param verbose: Whether to print progress messages. Default is True.
    :type verbose: bool, optional

    :returns: A tuple containing a DataFrame with normalised values and a DataFrame with model statistics.
    :rtype: tuple (pandas.DataFrame, pandas.DataFrame)

    Example:

    .. code-block:: python

        import pandas as pd
        import normet as nm
        df = pd.read_csv('timeseries_data.csv')
        value = 'target'
        feature_names = ['feature1', 'feature2', 'feature3']
        variables_resample = ['feature1', 'feature2']
        df_dew, mod_stats = nm.do_all_unc(df, value=value, feature_names=feature_names, variables_resample=variables_resample)

    Notes:

    - Multiple models are trained using different random seeds to quantify uncertainty.
    - If `verbose` is True, progress messages are printed.
    - normalisation is performed using the specified number of CPU cores, with the default being the total number of cores minus one.
    - If a weather DataFrame is provided, it is used for resampling meteorological parameters; otherwise, the input DataFrame is used.


.. function:: decom_emi(df=None, model=None, value=None, feature_names=None, split_method='random', fraction=0.75, model_config=None, n_samples=300, seed=7654321, n_cores=None, verbose=True)

    Decomposes a time series into different components using machine learning models.

    This function prepares the data, trains a machine learning model using AutoML, and decomposes the time series data into various components. The decomposition is based on the contribution of different features to the target variable. It returns the decomposed data and model statistics.

    :param df: Input dataframe containing the time series data.
    :type df: pandas.DataFrame
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
    :param n_samples: Number of samples for normalisation. Default is 300.
    :type n_samples: int, optional
    :param seed: Random seed for reproducibility. Default is 7654321.
    :type seed: int, optional
    :param n_cores: Number of cores to be used. Default is total CPU cores minus one.
    :type n_cores: int, optional
    :param verbose: Whether to print progress messages. Default is True.
    :type verbose: bool, optional
    :returns: A tuple containing a dataframe with decomposed components and a dataframe with model statistics.
    :rtype: tuple (pd.DataFrame, pd.DataFrame)

    **Example:**

    .. code-block:: python

        import pandas as pd
        import normet as nm
        df = pd.read_csv('timeseries_data.csv')
        value = 'target'
        feature_names = ['feature1', 'feature2', 'feature3']
        df_dewc, mod_stats = nm.decom_emi(df, value, feature_names)

    **Details:**

    - If no pre-trained model is provided, the function will prepare the data and train a new model using AutoML.
    - The function gathers model statistics for testing, training, and the entire dataset.
    - The time series is decomposed by excluding different features iteratively.
    - The decomposed components are adjusted to create deweathered values.
    - The results include the decomposed dataframe and model statistics for further analysis.


.. function:: decom_met(df=None, model=None, value=None, feature_names=None, split_method='random', fraction=0.75, model_config=None, n_samples=300, seed=7654321, importance_ascending=False, n_cores=None, verbose=True)

    Decomposes a time series into different components using machine learning models with feature importance ranking.

    This function prepares the data, trains a machine learning model using AutoML, and decomposes the time series data into various components. The decomposition is based on the feature importance ranking and their contributions to the target variable. It returns the decomposed data and model statistics.

    :param df: Input dataframe containing the time series data.
    :type df: pandas.DataFrame
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
    :param n_samples: Number of samples for normalisation. Default is 300.
    :type n_samples: int, optional
    :param seed: Random seed for reproducibility. Default is 7654321.
    :type seed: int, optional
    :param importance_ascending: Sort order for feature importances. Default is False.
    :type importance_ascending: bool, optional
    :param n_cores: Number of cores to be used. Default is total CPU cores minus one.
    :type n_cores: int, optional
    :param verbose: Whether to print progress messages. Default is True.
    :type verbose: bool, optional
    :returns: A dataframe with decomposed components and a dataframe with model statistics.
    :rtype: tuple (pd.DataFrame, pd.DataFrame)

    **Example:**

    .. code-block:: python

        import pandas as pd
        import normet as nm
        df = pd.read_csv('timeseries_data.csv')
        value = 'target'
        feature_names = ['feature1', 'feature2', 'feature3']
        df_dewwc, mod_stats = nm.decom_met(df, value, feature_names)

    **Details:**

    - If no pre-trained model is provided, the function will prepare the data and train a new model using AutoML.
    - The function gathers model statistics for testing, training, and the entire dataset.
    - Feature importances are determined and sorted based on their contribution to the target variable.
    - The time series is decomposed by excluding different features iteratively, according to their importance.
    - The decomposed components are adjusted to create weather-independent values.
    - The results include the decomposed dataframe and model statistics for further analysis.


.. function:: rolling_dew(df=None, model=None, value=None, feature_names=None, split_method='random', fraction=0.75, model_config=None, n_samples=300, window_days=14, rolling_every=7, seed=7654321, n_cores=None, verbose=True)

    Applies a rolling window approach to decompose the time series into different components using machine learning models.

    This function prepares the data, trains a machine learning model using AutoML, and applies a rolling window approach
    to decompose the time series data into various components. The decomposition is based on the contribution of different
    features to the target variable. It returns the decomposed data and model statistics.

    :param df: Input dataframe containing the time series data.
    :type df: pandas.DataFrame
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
    :param n_samples: Number of samples for normalisation. Default is 300.
    :type n_samples: int, optional
    :param window_days: Number of days for the rolling window. Default is 14.
    :type window_days: int, optional
    :param rolling_every: Rolling interval in days. Default is 7.
    :type rolling_every: int, optional
    :param seed: Random seed for reproducibility. Default is 7654321.
    :type seed: int, optional
    :param n_cores: Number of cores to be used. Default is total CPU cores minus one.
    :type n_cores: int, optional
    :param verbose: Whether to print progress messages. Default is True.
    :type verbose: bool, optional
    :returns: Tuple containing:
              - df_dew (pd.DataFrame): Dataframe with decomposed components including mean and standard deviation of the rolling window.
              - mod_stats (pd.DataFrame): Dataframe with model statistics.

    **Details:**

    - Data Preparation: Prepares the input data for modeling and optionally trains a new model using AutoML.
    - Model Training: Trains or uses the provided model to learn the relationship between features and the target variable.
    - Rolling Window Decomposition: Applies a rolling window approach to decompose the time series into components over specified windows and intervals.
    - Feature normalisation: Normalises the data within each rolling window using `normalise` function.
    - Component Calculation: Calculates mean and standard deviation of the rolling window to derive short-term and seasonal components.
    - Returns decomposed data (`df_dew`) including observed, short-term, seasonal components, and statistics (`mod_stats`) for evaluation.

    **Example:**

    - Useful for analyzing time series data with varying patterns over time and decomposing it into interpretable components.
    - Supports dynamic assessment of feature contributions to the target variable across different rolling windows.

    .. code-block:: python

        import pandas as pd
        import normet as nm
        df = pd.read_csv('timeseries_data.csv')
        value = 'target'
        feature_names = ['feature1', 'feature2', 'feature3']
        df_dew, mod_stats = nm.rolling_dew(df, value, feature_names, window_days=14, rolling_every=2)

    **Notes:**

    - Enhances understanding of time series data by breaking down its components over sliding windows.
    - Facilitates evaluation of model performance and feature relevance across different temporal contexts.


.. function:: modStats(df, model, set=None, statistic=None)

    Calculates statistics for model evaluation based on provided data.

    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param model: Trained ML model.
    :type model: object
    :param set: Set type for which statistics are calculated ('training', 'testing', or 'all'). Default is None.
    :type set: str, optional
    :param statistic: List of statistics to calculate. Default is ["n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA", "R2"].
    :type statistic: list of str, optional
    :return: DataFrame containing calculated statistics.
    :rtype: pandas.DataFrame

    **Example:**

    Calculates statistics for a trained model on testing dataset:

    .. code-block:: python

        import pandas as pd
        import normet as nm
        df = pd.read_csv('timeseries_data.csv')
        model = nm.train_model(df, 'target', feature_names)
        stats = nm.modStats(df, model, set='testing')

    **Notes:**

    - If `set` parameter is provided, the function filters the DataFrame `df` to include only rows where the 'set' column matches `set`.
    - Raises a ValueError if `set` parameter is provided but 'set' column is not present in `df`.
    - Calculates statistics such as 'n', 'FAC2', 'MB', 'MGE', 'NMB', 'NMGE', 'RMSE', 'r', 'COE', 'IOA', 'R2' based on model predictions ('value_predict') and observed values ('value') in the DataFrame.


.. function:: Stats(df, mod, obs, statistic=None)

    Calculates specified statistics based on provided data.

    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param mod: Column name of the model predictions.
    :type mod: str
    :param obs: Column name of the observed values.
    :type obs: str
    :param statistic: List of statistics to calculate. Default is ["n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA", "R2"].
    :type statistic: list of str, optional
    :returns: DataFrame containing calculated statistics.
    :rtype: pandas.DataFrame

    **Details:**

    This function calculates a range of statistical metrics to evaluate the model predictions against the observed values. The following statistics can be calculated:

    - **n**: Number of observations.
    - **FAC2**: Factor of 2.
    - **MB**: Mean Bias.
    - **MGE**: Mean Gross Error.
    - **NMB**: Normalised Mean Bias.
    - **NMGE**: Normalised Mean Gross Error.
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

    **Example:**

    .. code-block:: python

        import pandas as pd
        import normet as nm
        data = {
                 'observed': [1, 2, 3, 4, 5],
                 'predicted': [1.1, 1.9, 3.2, 3.8, 5.1]
         }
        df = pd.DataFrame(data)
        stats = nm.Stats(df, mod='predicted', obs='observed')
        print(stats)

    **Notes:**

    - Each statistical metric has a specific function that calculates its value.
    - The function returns a DataFrame with the calculated statistics.
    - Significance levels for the correlation coefficient are marked with appropriate symbols.


.. function:: pdp(df, model, variables=None, training_only=True, n_cores=None)

    Computes partial dependence plots for all specified features.

    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param model: AutoML model object.
    :param variables: List of variables to compute partial dependence plots for. If None, defaults to feature_names.
    :type variables: list, optional
    :param training_only: If True, computes partial dependence plots only for the training set. Default is True.
    :type training_only: bool, optional
    :param n_cores: Number of CPU cores to use. Default is total CPU cores minus one.
    :type n_cores: int, optional
    :return: DataFrame containing the computed partial dependence plots for all specified features.
    :rtype: pandas.DataFrame

    **Example:**

    .. code-block:: python

        import pandas as pd
        from flaml import AutoML
        import normet as nm  # Replace with the actual module name if different

        # Load dataset
        df = pd.read_csv('path_to_your_dataset.csv')

        # Initialize AutoML model
        automl = AutoML()

        # Fit the model (assuming the model has a fit method)
        automl.fit(df.drop(columns=['target']), df['target'])

        # Compute Partial Dependence Plots for specific features
        df_predict = nm.pdp(df, automl, variables=['feature1', 'feature2', 'feature3'])

        # Display the resulting DataFrame
        print(df_predict)


.. function:: pdp_worker(X_train, model, variable, training_only=True)

    Worker function for computing partial dependence plots for a single feature.

    :param model: AutoML model object.
    :param X_train: Input DataFrame containing the training data.
    :type X_train: pandas.DataFrame
    :param variable: Name of the feature to compute partial dependence plot for.
    :type variable: str
    :param training_only: If True, computes partial dependence plot only for the training set. Default is True.
    :type training_only: bool, optional
    :return: DataFrame containing the computed partial dependence plot for the specified feature.
    :rtype: pandas.DataFrame

    **Example:**

    .. code-block:: python

        import pandas as pd
        from flaml import AutoML
        import normet as nm

        # Load training data
        X_train = pd.read_csv('path_to_your_training_data.csv')

        # Initialize AutoML model
        automl = AutoML()

        # Fit the model (assuming the model has a fit method)
        automl.fit(X_train.drop(columns=['target']), X_train['target'])

        # Compute Partial Dependence Plot for a single feature
        df_predict = nm.pdp_worker(X_train, automl, variable='feature1')

        # Display the resulting DataFrame
        print(df_predict)



.. function:: scm_all(df, poll_col, code_col, control_pool, cutoff_date, n_cores=None)

    Performs Synthetic Control Method (SCM) in parallel for multiple treatment targets.

    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param poll_col: Name of the column containing the poll data.
    :type df: str
    :param code_col: Name of the column containing the code data.
    :type code_col: str
    :param control_pool: List of control pool codes.
    :type control_pool: list
    :param cutoff_date: Date for splitting pre- and post-treatment datasets.
    :type cutoff_date: str
    :param n_cores: Number of CPU cores to use. Default is total CPU cores minus one.
    :type n_cores: int, optional
    :return: DataFrame containing synthetic control results for all treatment targets.
    :rtype: pandas.DataFrame

    **Example:**

    .. code-block:: python

        import pandas as pd
        import normet as nm

        # Example data
        df = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
            'pollutant': np.random.randn(100),
            'unit_code': ['A'] * 25 + ['B'] * 25 + ['C'] * 25 + ['D'] * 25
        })

        # Perform SCM in parallel for multiple treatment targets
        synthetic_all = nm.scm_all(df, poll_col='pollutant', code_col='unit_code', control_pool=['B', 'C', 'D'], cutoff_date='2020-01-01', n_cores=4)


.. function:: scm(df, poll_col, code_col, treat_target, control_pool, cutoff_date)

    Performs Synthetic Control Method (SCM) for a single treatment target.

    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param poll_col: Name of the column containing the poll data.
    :type df: str
    :param code_col: Name of the column containing the code data.
    :type code_col: str
    :param treat_target: Code of the treatment target.
    :type treat_target: str
    :param control_pool: List of control pool codes.
    :type control_pool: list
    :param cutoff_date: Date for splitting pre- and post-treatment datasets.
    :type cutoff_date: str
    :return: DataFrame containing synthetic control results for the specified treatment target.
    :rtype: pandas.DataFrame

    **Example:**

    .. code-block:: python

        import pandas as pd
        import normet as nm

        # Example data
        df = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
            'pollutant': np.random.randn(100),
            'unit_code': ['A'] * 25 + ['B'] * 25 + ['C'] * 25 + ['D'] * 25
        })

        # Perform SCM for a single treatment target
        synthetic_data = nm.scm(df, poll_col='pollutant', code_col='unit_code', treat_target='A', control_pool=['B', 'C', 'D'], cutoff_date='2020-01-01')


.. function:: mlsc(df, poll_col, date_col, code_col, treat_target, control_pool, cutoff_date, model_config)

    Performs synthetic control using machine learning regression models.

    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param poll_col: Name of the column containing the poll data.
    :type poll_col: str
    :param date_col: Name of the column containing the date data.
    :type date_col: str
    :param code_col: Name of the column containing the code data.
    :type code_col: str
    :param treat_target: Code of the treatment target.
    :type treat_target: str
    :param control_pool: List of control pool codes.
    :type control_pool: list
    :param cutoff_date: Date for splitting pre- and post-treatment datasets.
    :type cutoff_date: str
    :param model_config: Configuration dictionary for model training parameters.
    :type model_config: dict, optional
    :return: DataFrame containing synthetic control results for the specified treatment target.
    :rtype: pandas.DataFrame

    **Example:**

    .. code-block:: python

        import normet as nm
        synthetic_result = nm.mlsc(df, poll_col='poll', date_col='date', code_col='code', treat_target='X', control_pool=['A', 'B', 'C'], cutoff_date='2020-01-01')

    **Notes:**

    - The default `model_config` includes:

    .. code-block:: python

        model_config = {
        'time_budget': 90,                     # Total running time in seconds
        'metric': 'r2',                        # Primary metric for regression
        'estimator_list': ["lgbm"],            # List of ML learners: "lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"
        'task': 'regression',                  # Task type
        'verbose': verbose                     # Print progress messages
        }

    - This configuration can be updated with user-provided `model_config`.



.. function:: mlsc_all(df, poll_col, date_col, code_col, control_pool, cutoff_date, training_time=60, n_cores=None)

    Performs synthetic control using machine learning regression models in parallel for multiple treatment targets.

    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param poll_col: Name of the column containing the poll data.
    :type poll_col: str
    :param date_col: Name of the column containing the date data.
    :type date_col: str
    :param code_col: Name of the column containing the code data.
    :type code_col: str
    :param control_pool: List of control pool codes.
    :type control_pool: list
    :param cutoff_date: Date for splitting pre- and post-treatment datasets.
    :type cutoff_date: str
    :param training_time: Total running time in seconds for the AutoML model. Default is 60.
    :type training_time: int, optional
    :param n_cores: Number of CPU cores to use. Default is total CPU cores minus one.
    :type n_cores: int, optional
    :return: DataFrame containing synthetic control results for all treatment targets.
    :rtype: pandas.DataFrame

    **Example:**

    .. code-block:: python

        import normet as nm
        synthetic_results = nm.mlsc_all(df, poll_col='poll', date_col='date', code_col='code', control_pool=['A', 'B', 'C'], cutoff_date='2020-01-01', training_time=60)
