import pandas as pd
import numpy as np
from datetime import datetime
from random import sample
from scipy import stats
from flaml import AutoML
from joblib import Parallel, delayed
import statsmodels.api as sm
from sklearn.inspection import partial_dependence
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import os
import time


def prepare_data(df, value, feature_names, na_rm=True, split_method='random', replace=False, fraction=0.75, seed=7654321):
    """
    Prepares the input DataFrame by performing data cleaning, imputation, and splitting.

    Parameters:
        df (pandas.DataFrame):: Input DataFrame containing the dataset.
        value (str): Name of the target variable.
        feature_names (list): List of feature names.
        na_rm (bool, optional): Whether to remove missing values. Default is True.
        split_method (str, optional): Method for splitting data ('random' or 'time_series'). Default is 'random'.
        replace (bool, optional): Whether to replace existing date variables. Default is False.
        fraction (float, optional): Fraction of the dataset to be used for training. Default is 0.75.
        seed (int, optional): Seed for random operations. Default is 7654321.

    Returns:
        DataFrame: Prepared DataFrame with cleaned data and split into training and testing sets.
    """

    # Perform the data preparation steps
    df = (df
            .pipe(process_df, variables_col = feature_names + [value])
            .pipe(check_data, value = value)
            .pipe(impute_values, na_rm = na_rm)
            .pipe(add_date_variables, replace = replace)
            .pipe(split_into_sets, split_method = split_method, fraction = fraction, seed = seed)
            .reset_index(drop = True))

    return df


def process_df(df, variables_col):
    """
    Processes the DataFrame to ensure it contains necessary date and selected feature columns.

    This function checks if the date is present in the index or columns, selects the necessary features and
    the date column, and prepares the DataFrame for further analysis.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.
        variables_col (list of str): List of variable names to be included in the DataFrame.

    Returns:
        pd.DataFrame: Processed DataFrame containing the date and selected feature columns.

    Raises:
        ValueError: If no datetime information is found in index or columns.
    """
    # Check if the date is in the index or columns
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    time_column = df.select_dtypes(include = 'datetime64').columns

    if len(time_column) == 0:
        raise ValueError("No datetime information found in index or columns.")
    elif len(time_column) > 1:
        raise ValueError("More than one datetime column found.")

    # Select features and the date column
    selected_columns = list(set(variables_col) & set(df.columns))
    selected_columns.append(time_column[0])

    df = df[selected_columns].rename(columns = {time_column[0]: 'date'})

    return df


def check_data(df, value):
    """
    Validates and preprocesses the input DataFrame for subsequent analysis or modeling.

    This function checks if the target variable is present, ensures the date column is of the correct type,
    and validates there are no missing dates, returning a DataFrame with the target column renamed for consistency.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing the data to be checked.
        value (str): Name of the target variable (column) to be used in the analysis.

    Returns:
        pd.DataFrame: A DataFrame containing only the necessary columns, with appropriate checks and transformations applied.

    Raises:
        ValueError: If any of the following conditions are met:
            - The target variable (`value`) is not in the DataFrame columns.
            - There is no datetime information in either the index or the 'date' column.
            - The 'date' column is not of type datetime64.
            - The 'date' column contains missing values.

    Notes:
        - If the DataFrame's index is a DatetimeIndex, it is reset to a column named 'date'.
        - The target column (`value`) is renamed to 'value'.

    Example:
        >>> data = {
        ...     'timestamp': pd.date_range(start='1/1/2020', periods=5, freq='D'),
        ...     'target': [1, 2, 3, 4, 5]
        ... }
        >>> df = pd.DataFrame(data).set_index('timestamp')
        >>> df_checked = check_data(df, 'target')
        >>> print(df_checked)
    """

    # Check if the target variable is in the DataFrame
    if value not in df.columns:
        raise ValueError(f"The target variable `{value}` is not in the DataFrame columns.")

    # Rename the target column to 'value'
    df = df.rename(columns={value: "value"})

    # Check if the date column is of type datetime64
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        raise ValueError("`date` variable needs to be a parsed date (datetime64).")

    # Check if the date column contains any missing values
    if df['date'].isnull().any():
        raise ValueError("`date` must not contain missing (NA) values.")

    return df


def impute_values(df, na_rm):
    """
    Imputes missing values in the DataFrame.

    Parameters:
        df (pandas.DataFrame):: Input DataFrame containing the dataset.
        na_rm (bool): Whether to remove missing values.

    Returns:
        DataFrame: DataFrame with imputed missing values.
    """
    # Remove missing values if na_rm is True
    if na_rm:
        df = df.dropna(subset=['value']).reset_index(drop=True)

    # Impute missing values for numeric variables
    for col in df.select_dtypes(include=[np.number]).columns:
        df.fillna({col: df[col].median()}, inplace=True)

    # Impute missing values for character and categorical variables
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df.fillna({col: df[col].mode()[0]}, inplace=True)

    return df


def add_date_variables(df, replace):
    """
    Adds date-related variables to the DataFrame.

    Parameters:
        df (pandas.DataFrame):: Input DataFrame containing the dataset.
        replace (bool): Whether to replace existing date variables.

    Returns:
        DataFrame: DataFrame with added date-related variables.
    """
    if replace:
        # Replace existing date-related variables if they exist
        df['date_unix'] = df['date'].astype(np.int64) // 10**9
        df['day_julian'] = pd.DatetimeIndex(df['date']).dayofyear
        df['weekday'] = pd.DatetimeIndex(df['date']).weekday + 1
        df['weekday'] = df['weekday'].astype("category")
        df['hour'] = pd.DatetimeIndex(df['date']).hour
    else:
        # Add date-related variables only if they don't already exist
        if 'date_unix' not in df.columns:
            df['date_unix'] = df['date'].apply(lambda x: x.timestamp())
        if 'day_julian' not in df.columns:
            df['day_julian'] = df['date'].apply(lambda x: x.timetuple().tm_yday)
        if 'weekday' not in df.columns:
            df['weekday'] = df['date'].apply(lambda x: x.weekday() + 1)
            df['weekday'] = df['weekday'].astype("category")
        if 'hour' not in df.columns:
            df['hour'] = df['date'].apply(lambda x: x.hour)

    return df


def split_into_sets(df, split_method, fraction, seed):
    """
    Splits the DataFrame into training and testing sets.

    Parameters:
        df (pandas.DataFrame):: Input DataFrame containing the dataset.
        split_method (str): Method for splitting data ('random', 'ts', 'season', 'month').
        fraction (float): Fraction of the dataset to be used for training (for 'random', 'ts', 'season')
                          or fraction of each month to be used for training (for 'month').
        seed (int): Seed for random operations.

    Returns:
        DataFrame: DataFrame with a 'set' column indicating the training or testing set.
    """
    # Add row number
    df = df.reset_index().rename(columns={'index': 'rowid'})

    if split_method == 'random':
        # Sample to get training set
        df_training = df.sample(frac=fraction, random_state=seed).reset_index(drop=True).assign(set="training")
        # Remove training set from input to get testing set
        df_testing = df[~df['rowid'].isin(df_training['rowid'])].assign(set="testing")

    elif split_method == 'ts':
        # Time series split
        split_index = int(fraction * len(df))
        df_training = df.iloc[:split_index].reset_index(drop=True).assign(set="training")
        df_testing = df.iloc[split_index:].reset_index(drop=True).assign(set="testing")

    elif split_method == 'season':
        # Function to map month to season
        def get_season(month):
            if month in [12, 1, 2]:
                return 'DJF'
            elif month in [3, 4, 5]:
                return 'MAM'
            elif month in [6, 7, 8]:
                return 'JJA'
            else:
                return 'SON'

        df['season'] = df['date'].dt.month.apply(get_season)

        df_training_list = []
        df_testing_list = []

        for season in ['DJF', 'MAM', 'JJA', 'SON']:
            season_df = df[df['season'] == season]
            split_index = int(fraction * len(season_df))
            season_training = season_df.iloc[:split_index].reset_index(drop=True).assign(set="training")
            season_testing = season_df.iloc[split_index:].reset_index(drop=True).assign(set="testing")

            df_training_list.append(season_training)
            df_testing_list.append(season_testing)

        df_training = pd.concat(df_training_list).reset_index(drop=True)
        df_testing = pd.concat(df_testing_list).reset_index(drop=True)

    elif split_method == 'month':
        # Extract month from date column
        df['month'] = df['date'].dt.month

        df_training_list = []
        df_testing_list = []

        # Iterate over each month
        for month in range(1, 13):
            month_df = df[df['month'] == month]
            split_index = int(fraction * len(month_df))
            month_training = month_df.iloc[:split_index].reset_index(drop=True).assign(set="training")
            month_testing = month_df.iloc[split_index:].reset_index(drop=True).assign(set="testing")

            df_training_list.append(month_training)
            df_testing_list.append(month_testing)

        df_training = pd.concat(df_training_list).reset_index(drop=True)
        df_testing = pd.concat(df_testing_list).reset_index(drop=True)

    # Combine training and testing sets
    df_split = pd.concat([df_training, df_testing]).sort_values(by='date').reset_index(drop=True)

    return df_split


def train_model(df, value='value', variables=None, model_config=None, seed=7654321, verbose=True):
    """
    Trains a machine learning model using the provided dataset and parameters.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing the dataset.
        value (str, optional): Name of the target variable. Default is 'value'.
        variables (list of str, optional): List of feature variables. Default is None.

    Keyword Parameters:
        model_config (dict, optional): Configuration dictionary for model training parameters.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        verbose (bool, optional): If True, print progress messages. Default is True.

    Returns:
        object: Trained ML model object.

    Raises:
        ValueError: If `variables` contains duplicates or if any `variables` are not present in the DataFrame.
    """

    # Check for duplicate variables
    if len(set(variables)) != len(variables):
        raise ValueError("`variables` contains duplicate elements.")

    # Check if all variables are in the DataFrame
    if not all(var in df.columns for var in variables):
        raise ValueError("`variables` given are not within input data frame.")

    # Extract relevant data for training
    if 'set' in df.columns:
        df_train = df[df['set'] == 'training'][[value] + variables]
    else:
        df_train = df[[value] + variables]

    # Default configuration for model training
    default_model_config = {
        'time_budget': 60,                     # Total running time in seconds
        'metric': 'rmse',                      # Primary metric for regression
        'estimator_list': [
            "lgbm", "rf", "xgboost",
            "extra_tree", "xgb_limitdepth"
        ],                                     # List of ML learners
        'task': 'regression',                  # Task type
        'verbose': verbose                     # Print progress messages
    }

    # Update default configuration with user-provided config
    if model_config is not None:
        default_model_config.update(model_config)

    # Initialize and train AutoML model
    model = AutoML()
    if verbose:
        print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), ": Training AutoML...")

    model.fit(X_train=df_train[variables], y_train=df_train[value],
                **default_model_config, seed=seed)

    if verbose:
        print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), ": Best model is",
            model.best_estimator, "with best model parameters of", model.best_config)

    return model


def prepare_train_model(df, value, feature_names, split_method, fraction, model_config, seed, verbose=True):
    """
    Prepares the data and trains a machine learning model using the specified configuration.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data to be used for training.
        value (str): The name of the target variable to be predicted.
        feature_names (list of str): A list of feature column names to be used in the training.
        split_method (str): The method to split the data ('random' or other supported methods).
        fraction (float): The fraction of data to be used for training.
        model_config (dict): The configuration dictionary for the AutoML model training.
        seed (int): The random seed for reproducibility.
        verbose (bool, optional): If True, print progress messages. Default is True.

    Returns:
        tuple:
            - pd.DataFrame: The prepared DataFrame ready for model training.
            - object: The trained machine learning model.

    Raises:
        ValueError: If there are any issues with the data preparation or model training.

    Example:
        >>> data = {
        ...     'feature1': [1, 2, 3, 4, 5],
        ...     'feature2': [5, 4, 3, 2, 1],
        ...     'target': [2, 3, 4, 5, 6],
        ...     'set': ['training', 'training', 'training', 'testing', 'testing']
        ... }
        >>> df = pd.DataFrame(data)
        >>> feature_names = ['feature1', 'feature2']
        >>> split_method = 'random'
        >>> fraction = 0.75
        >>> model_config = {'time_budget': 60, 'metric': 'rmse'}
        >>> seed = 7654321
        >>> df_prepared, model = prepare_train_model(df, value='target', feature_names=feature_names, split_method=split_method, fraction=fraction, model_config=model_config, seed=seed, verbose=True)
    """

    vars = list(set(feature_names) - set(['date_unix', 'day_julian', 'weekday', 'hour']))

    # Prepare the data
    df = prepare_data(df, value=value, feature_names=vars, split_method=split_method, fraction=fraction, seed=seed)

    # Train the model using AutoML
    model = train_model(df, value='value', variables=feature_names, model_config=model_config, seed=seed, verbose=verbose)

    return df, model


def normalise_worker(index, df, model, variables_resample, replace, seed, verbose, weather_df=None):
    """
    Worker function for parallel normalisation of data using randomly resampled meteorological parameters
    from another weather DataFrame within its date range. If no weather DataFrame is provided,
    it defaults to using the input DataFrame.

    Parameters:
        index (int): Index of the worker.
        df (pandas.DataFrame): Input DataFrame containing the dataset.
        model (ML): Trained ML model.
        variables_resample (list of str): List of resampling variables.
        replace (bool): Whether to sample with replacement.
        seed (int): Random seed.
        verbose (bool): Whether to print progress messages.
        weather_df (pandas.DataFrame, optional): Weather DataFrame containing the meteorological parameters.
                                             Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing normalised predictions.
    """

    # Print progress message every fifth prediction
    if verbose and index % 5 == 0:
        # Calculate and format the progress percentage
        message_percent = round((index / len(df)) * 100, 2)
        message_percent = "{:.1f} %".format(message_percent)
        print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
              ": Predicting", index, "of", len(df), "times (", message_percent, ")...")

    # Randomly sample observations within the weather DataFrame
    np.random.seed(seed)

    # Sample meteorological parameters from weather_df
    sampled_meteorological_params = weather_df[variables_resample].sample(n=len(weather_df), replace=replace).reset_index(drop=True)

    # Use the sampled parameters directly (without modifying df)
    df[variables_resample] = sampled_meteorological_params.sample(n=len(df), replace=True).reset_index(drop=True)

    # Predict using the model
    value_predict = model.predict(df)

    # Build DataFrame of predictions
    predictions = pd.DataFrame({
        'date': df['date'],
        'observed': df['value'],
        'normalised': value_predict,
        'seed' : seed
    })

    return predictions


def normalise(df, model, feature_names, variables_resample=None, n_samples=300, replace=True,
              aggregate=True, seed=7654321, n_cores=None, weather_df=None, verbose=True):
    """
    Normalises the dataset using the trained model.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing the dataset.
        model (object): Trained ML model.
        feature_names (list of str): List of feature names.
        variables_resample (list of str): List of resampling variables.
        n_samples (int, optional): Number of samples to normalise. Default is 300.
        replace (bool, optional): Whether to replace existing data. Default is True.
        aggregate (bool, optional): Whether to aggregate results. Default is True.
        seed (int, optional): Random seed. Default is 7654321.
        n_cores (int, optional): Number of CPU cores to use. Default is total CPU cores minus one.
        weather_df (pandas.DataFrame, optional): DataFrame containing weather data for resampling. Default is None.
        verbose (bool, optional): Whether to print progress messages. Default is True.

    Returns:
        pd.DataFrame: DataFrame containing normalised predictions.

    Example:
        >>> data = {
        ...     'date': pd.date_range(start='2020-01-01', periods=5, freq='D'),
        ...     'feature1': [1, 2, 3, 4, 5],
        ...     'feature2': [5, 4, 3, 2, 1],
        ...     'value': [2, 3, 4, 5, 6]
        ... }
        >>> df = pd.DataFrame(data)
        >>> feature_names = ['feature1', 'feature2']
        >>> model = train_model(df, value='value', variables=feature_names)
        >>> variables_resample = ['feature1', 'feature2']
        >>> normalised_df = normalise(df, model, feature_names, variables_resample)
    """

    # Process input DataFrames
    df = process_df(df, variables_col=feature_names + ['value'])

    # If no weather_df is provided, use df as the weather data
    if weather_df is None:
        weather_df = df

    # Use all variables except the trend term
    if variables_resample is None:
        variables_resample = [var for var in feature_names if var != 'date_unix']

    # Check if all variables are in the DataFrame
    if not all(var in weather_df.columns for var in variables_resample):
        raise ValueError("The input weather_df does not contain all variables within `variables_resample`.")

    # Generate random seeds for parallel processing
    np.random.seed(seed)
    random_seeds = np.random.choice(np.arange(1000001), size=n_samples, replace=False)

    # Determine number of CPU cores to use
    n_cores = n_cores if n_cores is not None else os.cpu_count() - 1

    if verbose:
        print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), ": Normalising the dataset using the trained model in parallel.")

    # Perform normalisation using parallel processing
    df_result = pd.concat(Parallel(n_jobs=n_cores)(delayed(normalise_worker)(
            index=i, df=df, model=model, variables_resample=variables_resample, replace=replace,
            seed=random_seeds[i], verbose=False, weather_df=weather_df) for i in range(n_samples)), axis=0)

    # Aggregate results if needed
    if aggregate:
        if verbose:
            print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), ": Aggregating", n_samples, "predictions...")
        df_result = df_result.pivot_table(index='date', aggfunc='mean')[['observed', 'normalised']]
    else:
        # Pivot table to reshape 'normalised' values by 'seed' and set 'date' as index
        normalised_pivot = df_result.pivot_table(index='date', columns='seed', values='normalised')

        # Select and drop duplicate rows based on 'date', keeping only 'observed' column
        observed_unique = df_result[['date', 'observed']].drop_duplicates().set_index('date')

        # Concatenate the pivoted 'normalised' values and unique 'observed' values
        df_result = pd.concat([observed_unique, normalised_pivot], axis=1)
        if verbose:
            print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), ": Concatenated", n_samples, "predictions...")

    return df_result


def do_all(df=None, model=None, value=None, feature_names=None, variables_resample=None, split_method='random', fraction=0.75,
           model_config=None, n_samples=300, seed=7654321, n_cores=None, aggregate=True, weather_df=None, verbose=True):
    """
    Conducts data preparation, model training, and normalisation, returning the transformed dataset and model statistics.

    This function performs the entire pipeline from data preparation to model training and normalisation using
    specified parameters and returns the transformed dataset along with model statistics.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing the dataset.
        model (object, optional): Pre-trained model to use for decomposition. If None, a new model will be trained. Default is None.
        value (str): Name of the target variable.
        feature_names (list of str): List of feature names.
        variables_resample (list of str): List of variables for normalisation.
        split_method (str, optional): Method for splitting data ('random' or 'time_series'). Default is 'random'.
        fraction (float, optional): Fraction of the dataset to be used for training. Default is 0.75.
        model_config (dict, optional): Configuration dictionary for model training parameters.
        n_samples (int, optional): Number of samples for normalisation. Default is 300.
        seed (int, optional): Seed for random operations. Default is 7654321.
        n_cores (int, optional): Number of CPU cores to be used for normalisation. Default is total CPU cores minus one.
        weather_df (pandas.DataFrame, optional): DataFrame containing weather data for resampling. Default is None.
        verbose (bool, optional): Whether to print progress messages. Default is True.

    Returns:
        tuple:
            - df_dew (pandas.DataFrame): Transformed dataset with normalised values.
            - mod_stats (pandas.DataFrame): DataFrame containing model statistics.

    Example:
        >>> df = pd.read_csv('timeseries_data.csv')
        >>> value = 'target'
        >>> feature_names = ['feature1', 'feature2', 'feature3']
        >>> variables_resample = ['feature1', 'feature2']
        >>> df_dew, mod_stats = do_all(df, value, feature_names, variables_resample)
    """
    # Train model if not provided
    if model is None:
        df, model= prepare_train_model(df, value, feature_names, split_method, fraction, model_config, seed, verbose)

    # Collect model statistics
    mod_stats = pd.concat([
        modStats(df, model, set='testing'),
        modStats(df, model, set='training'),
        modStats(df.assign(set="all"), model, set='all')], ignore_index=True)

    # Default logic for cpu cores
    n_cores = n_cores if n_cores is not None else os.cpu_count() - 1

    # Normalise the data using weather_df if provided
    df_dew = normalise(df, model, feature_names=feature_names, variables_resample=variables_resample, n_samples=n_samples,
                       aggregate=aggregate, n_cores=n_cores, seed=seed, weather_df=weather_df, verbose=verbose)

    return df_dew, mod_stats


def do_all_unc(df=None, value=None, feature_names=None, variables_resample=None, split_method='random', fraction=0.75,
               model_config=None, n_samples=300, n_models=10, confidence_level=0.95, seed=7654321, n_cores=None, weather_df=None, verbose=True):
    """
    Performs uncertainty quantification by training multiple models with different random seeds and calculates statistical metrics.

    Parameters:
        df (pandas.DataFrame): Input dataframe containing the time series data.
        value (str): Column name of the target variable.
        feature_names (list of str): List of feature column names.
        variables_resample (list of str): List of sampled feature names for normalisation.
        split_method (str, optional): Method to split the data ('random' or other methods). Default is 'random'.
        fraction (float, optional): Fraction of data to be used for training. Default is 0.75.
        model_config (dict, optional): Configuration dictionary for model training parameters.
        n_samples (int, optional): Number of samples for normalisation. Default is 300.
        n_models (int, optional): Number of models to train for uncertainty quantification. Default is 10.
        confidence_level (float, optional): Confidence level for the uncertainty bounds. Default is 0.95.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        n_cores (int, optional): Number of cores to be used. Default is total CPU cores minus one.
        weather_df (pandas.DataFrame, optional): DataFrame containing weather data for resampling. Default is None.
        verbose (bool, optional): Whether to print progress messages. Default is True.

    Returns:
        tuple:
            - df_dew (pandas.DataFrame): Dataframe with observed values, mean, standard deviation, median, lower and upper bounds, and weighted values.
            - mod_stats (pandas.DataFrame): Dataframe with model statistics.
    """

    np.random.seed(seed)
    random_seeds = np.random.choice(np.arange(1000001), size=n_models, replace=False)

    df_dew_list = []
    mod_stats_list = []

    # Determine number of CPU cores to use
    n_cores = n_cores if n_cores is not None else os.cpu_count() - 1

    start_time = time.time()  # Record start time for ETA calculation

    for i, seed in enumerate(random_seeds):
        df_dew0, mod_stats0 = do_all(df, value=value, feature_names=feature_names,
                                     variables_resample=variables_resample,
                                     split_method=split_method, fraction=fraction,
                                     model_config=model_config,
                                     n_samples=n_samples, seed=seed, n_cores=n_cores,
                                     weather_df=weather_df, verbose=False)

        df_dew0.rename(columns={'normalised': f'normalised_{seed}'}, inplace=True)
        df_dew0 = df_dew0[['observed', f'normalised_{seed}']]
        df_dew_list.append(df_dew0)

        mod_stats0['seed'] = seed
        mod_stats_list.append(mod_stats0)

        if verbose:
            elapsed_time = time.time() - start_time
            progress_percent = (i + 1) / n_models * 100

            # Calculate remaining time
            remaining_time = elapsed_time / (i + 1) * (n_models - (i + 1))

            # Format remaining time
            if remaining_time < 60:
                remaining_str = "ETA: {:.2f} seconds".format(remaining_time)
            elif remaining_time < 3600:
                remaining_str = "ETA: {:.2f} minutes".format(remaining_time / 60)
            else:
                remaining_str = "ETA: {:.2f} hours".format(remaining_time / 3600)

            print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                  ": Progress: {:.2f}% (Model {}/{})... {}".format(progress_percent, i + 1, n_models, remaining_str))

    df_dew = pd.concat(df_dew_list, axis=1)

    # Keep only the first 'observed' column and drop duplicates
    observed_columns = [col for col in df_dew.columns if 'observed' in col]
    df_dew = df_dew.loc[:, ~df_dew.columns.duplicated()]

    mod_stats = pd.concat(mod_stats_list, ignore_index=True)

    # Calculate statistics
    df_dew['mean'] = np.mean(df_dew.iloc[:, 1:n_models + 1], axis=1)
    df_dew['std'] = np.std(df_dew.iloc[:, 1:n_models + 1], axis=1)
    df_dew['median'] = np.median(df_dew.iloc[:, 1:n_models + 1], axis=1)
    df_dew['lower_bound'] = np.quantile(df_dew.iloc[:, 1:n_models + 1], (1 - confidence_level) / 2, axis=1)
    df_dew['upper_bound'] = np.quantile(df_dew.iloc[:, 1:n_models + 1], 1 - (1 - confidence_level) / 2, axis=1)

    # Calculate weighted R2
    test_stats = mod_stats[mod_stats['set'] == 'testing']
    test_stats.loc[:, 'R2'] = test_stats['R2'].replace([np.inf, -np.inf], np.nan)
    normalised_R2 = (test_stats['R2'] - test_stats['R2'].min()) / (test_stats['R2'].max() - test_stats['R2'].min())
    weighted_R2 = normalised_R2 / normalised_R2.sum()

    df_dew_weighted = df_dew.copy()
    df_dew_weighted.iloc[:, 1:n_models + 1] = (df_dew.iloc[:, 1:n_models + 1].values * weighted_R2.values[np.newaxis, :]).astype(np.float32)
    df_dew.loc[:,'weighted'] = df_dew_weighted.iloc[:, 1:n_models + 1].sum(axis=1)

    return df_dew, mod_stats


def decom_emi(df=None, model=None, value=None, feature_names=None, split_method='random', fraction=0.75,
             model_config=None, n_samples=300, seed=7654321, n_cores=None, verbose=True):
    """
    Decomposes a time series into different components using machine learning models.

    Parameters:
        df (pandas.DataFrame): Input dataframe containing the time series data.
        model (object, optional): Pre-trained model to use for decomposition. If None, a new model will be trained. Default is None.
        value (str): Column name of the target variable.
        feature_names (list of str): List of feature column names.
        split_method (str, optional): Method to split the data ('random' or other methods). Default is 'random'.
        fraction (float, optional): Fraction of data to be used for training. Default is 0.75.
        model_config (dict, optional): Configuration dictionary for model training parameters.
        n_samples (int, optional): Number of samples for normalisation. Default is 300.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        n_cores (int, optional): Number of cores to be used. Default is total CPU cores minus one.
        verbose (bool, optional): Whether to print progress messages. Default is True.

    Returns:
        tuple:
            - df_dewc (pandas.DataFrame): Dataframe with decomposed components.
            - mod_stats (pandas.DataFrame): Dataframe with model statistics.

    Example:
        >>> df = pd.read_csv('timeseries_data.csv')
        >>> value = 'target'
        >>> feature_names = ['feature1', 'feature2', 'feature3']
        >>> df_dewc, mod_stats = decom_emi(df, value, feature_names)
    """
    if model is None:
        df, model = prepare_train_model(df, value, feature_names, split_method, fraction, model_config, seed, verbose=True)

    # Gather model statistics for testing, training, and all data
    mod_stats = pd.concat([modStats(df, model, set='testing'),
                           modStats(df, model, set='training'),
                           modStats(df.assign(set="all"), model, set='all')], ignore_index=True)

    # Initialize the dataframe for decomposed components
    df_dew = df[['date', 'value']].set_index('date').rename(columns={'value': 'observed'})

    # Default logic for cpu cores
    n_cores = n_cores if n_cores is not None else os.cpu_count() - 1

    # Decompose the time series by excluding different features
    var_names = feature_names
    start_time = time.time()  # Initialize start time before the loop

    for i, var_to_exclude in enumerate(['base', 'date_unix', 'day_julian', 'weekday', 'hour']):
        if verbose:
            if i == 0:
                print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), f": Subtracting {var_to_exclude}...")
            else:
                elapsed_time = time.time() - start_time
                remaining_time = elapsed_time / i * (len(['base', 'date_unix', 'day_julian', 'weekday', 'hour']) - i)
                if remaining_time < 60:
                    eta_str = "ETA: {:.2f} seconds".format(remaining_time)
                elif remaining_time < 3600:
                    eta_str = "ETA: {:.2f} minutes".format(remaining_time / 60)
                else:
                    eta_str = "ETA: {:.2f} hours".format(remaining_time / 3600)
                print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), f": Subtracting {var_to_exclude}... {eta_str}")

        var_names = list(set(var_names) - set([var_to_exclude]))

        df_dew_temp = normalise(df, model, feature_names=feature_names, variables_resample=var_names,
                                n_samples=n_samples, n_cores=n_cores, seed=seed, verbose=False)

        df_dew[var_to_exclude] = df_dew_temp['normalised']

    # Adjust the decomposed components to create deweathered values
    df_dew['deweathered'] = df_dew['hour']
    df_dew['hour'] = df_dew['hour'] - df_dew['weekday']
    df_dew['weekday'] = df_dew['weekday'] - df_dew['day_julian']
    df_dew['day_julian'] = df_dew['day_julian'] - df_dew['date_unix']
    df_dew['date_unix'] = df_dew['date_unix'] - df_dew['base'] + df_dew['base'].mean()
    df_dew['emi_noise'] = df_dew['base'] - df_dew['base'].mean()

    return df_dew, mod_stats


def decom_met(df=None, model=None, value=None, feature_names=None, split_method='random', fraction=0.75,
                model_config=None, n_samples=300, seed=7654321, importance_ascending=False, n_cores=None, verbose=True):
    """
    Decomposes a time series into different components using machine learning models with feature importance ranking.

    Parameters:
        df (pandas.DataFrame): Input dataframe containing the time series data.
        model (object, optional): Pre-trained model to use for decomposition. If None, a new model will be trained. Default is None.
        value (str): Column name of the target variable.
        feature_names (list of str): List of feature column names.
        split_method (str, optional): Method to split the data ('random' or other methods). Default is 'random'.
        fraction (float, optional): Fraction of data to be used for training. Default is 0.75.
        model_config (dict, optional): Configuration dictionary for model training parameters.
        n_samples (int, optional): Number of samples for normalisation. Default is 300.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        importance_ascending (bool, optional): Sort order for feature importances. Default is False.
        n_cores (int, optional): Number of cores to be used. Default is total CPU cores minus one.
        verbose (bool, optional): Whether to print progress messages. Default is False.

    Returns:
        df_dewwc (pandas.DataFrame): Dataframe with decomposed components.
        mod_stats (pandas.DataFrame): Dataframe with model statistics.

    Example:
        >>> df = pd.read_csv('timeseries_data.csv')
        >>> value = 'target'
        >>> feature_names = ['feature1', 'feature2', 'feature3']
        >>> df_dewwc, mod_stats = decom_met(df, value, feature_names)
    """
    if model is None:
        df, model = prepare_train_model(df, value, feature_names, split_method, fraction, model_config, seed, verbose=True)

    # Gather model statistics for testing, training, and all data
    mod_stats = pd.concat([
        modStats(df, model, set='testing'),
        modStats(df, model, set='training'),
        modStats(df.assign(set="all"), model, set='all')], ignore_index=True)

    # Determine feature importances and sort them
    modelfi = pd.DataFrame(data={'feature_importances': model.feature_importances_},
                            index=model.feature_names_in_).sort_values('feature_importances', ascending=importance_ascending)

    # Initialize the dataframe for decomposed components
    df_deww = df[['date', 'value']].set_index('date').rename(columns={'value': 'observed'})
    met_list = ['deweathered'] + [item for item in modelfi.index if item not in ['hour', 'weekday', 'day_julian', 'date_unix']]
    var_names = [item for item in modelfi.index if item not in ['hour', 'weekday', 'day_julian', 'date_unix']]

    # Default logic for cpu cores
    n_cores = n_cores if n_cores is not None else os.cpu_count() - 1

    # Decompose the time series by excluding different features based on their importance
    start_time = time.time()  # Initialize start time before the loop
    for i, var_to_exclude in enumerate(met_list):
        if verbose:
            if i == 0:
                print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), f": Subtracting {var_to_exclude}...")
            else:
                elapsed_time = time.time() - start_time
                remaining_time = elapsed_time / i * (len(met_list) - i)
                if remaining_time < 60:
                    eta_str = "ETA: {:.2f} seconds".format(remaining_time)
                elif remaining_time < 3600:
                    eta_str = "ETA: {:.2f} minutes".format(remaining_time / 60)
                else:
                    eta_str = "ETA: {:.2f} hours".format(remaining_time / 3600)
                print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), f": Subtracting {var_to_exclude}... {eta_str}")

        var_names = list(set(var_names) - set([var_to_exclude]))

        df_dew_temp = normalise(df, model, feature_names=feature_names, variables_resample=var_names,
                                n_samples=n_samples, n_cores=n_cores, seed=seed, verbose=False)
        df_deww[var_to_exclude] = df_dew_temp['normalised']

    # Adjust the decomposed components to create weather-independent values
    df_dewwc = df_deww.copy()
    for i, param in enumerate([item for item in modelfi.index if item not in ['hour', 'weekday', 'day_julian', 'date_unix']]):
        if i > 0:
            df_dewwc[param] = df_deww[param] - df_deww[met_list[i - 1]]
        else:
            df_dewwc[param] = df_deww[param] - df_deww['deweathered']

    df_dewwc['met_noise'] = df_deww['observed'] - df_deww[met_list[-1]]

    return df_dewwc, mod_stats


def rolling_dew(df=None, model=None, value=None, feature_names=None, variables_resample=None, split_method='random',
                fraction=0.75, model_config=None, n_samples=300, window_days=14, rollingevery=7, seed=7654321, n_cores=None, verbose=True):
    """
    Applies a rolling window approach to decompose the time series into different components using machine learning models.

    Parameters:
        df (pandas.DataFrame): Input dataframe containing the time series data.
        model (object, optional): Pre-trained model to use for decomposition. If None, a new model will be trained. Default is None.
        value (str): Column name of the target variable.
        feature_names (list of str): List of feature column names.
        variables_resample (list of str): List of sampled feature names for normalisation.
        split_method (str, optional): Method to split the data ('random' or other methods). Default is 'random'.
        fraction (float, optional): Fraction of data to be used for training. Default is 0.75.
        model_config (dict, optional): Configuration dictionary for model training parameters.
        n_samples (int, optional): Number of samples for normalisation. Default is 300.
        window_days (int, optional): Number of days for the rolling window. Default is 14.
        rollingevery (int, optional): Rolling interval in days. Default is 7.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        n_cores (int, optional): Number of cores to be used. Default is total CPU cores minus one.
        verbose (bool, optional): Whether to print progress messages. Default is True.

    Returns:
        dfr (pandas.DataFrame): Dataframe with rolling decomposed components.
        mod_stats (pandas.DataFrame): Dataframe with model statistics.

    Example:
        >>> df = pd.read_csv('timeseries_data.csv')
        >>> value = 'target'
        >>> feature_names = ['feature1', 'feature2', 'feature3']
        >>> variables_resample = ['feature1', 'feature2']
        >>> dfr, mod_stats = rolling_dew(df, value, feature_names, variables_resample)
    """
    if model is None:
        df, model = prepare_train_model(df, value, feature_names, split_method, fraction, model_config, seed, verbose=True)

    # Collect model statistics
    mod_stats = pd.concat([
        modStats(df, model, set='testing'),
        modStats(df, model, set='training'),
        modStats(df.assign(set="all"), model, set='all')], ignore_index=True)

    # Create an initial dataframe to store observed values
    dfr = pd.DataFrame(index=df['date'], data={'observed': list(df['value'])})
    df['date_d'] = df['date'].dt.date

    # Define the rolling window range
    date_max = df['date_d'].max() - pd.DateOffset(days=window_days - 1)
    date_min = df['date_d'].min() + pd.DateOffset(days=window_days - 1)

    # Default logic for cpu cores
    n_cores = n_cores if n_cores is not None else os.cpu_count() - 1

    # Iterate over the rolling windows
    for i, ds in enumerate(pd.to_datetime(df['date_d'][df['date_d'] <= date_max.date()]).unique()[::rollingevery]):

        dfa = df[df['date_d'] >= ds.date()]
        dfa = dfa[dfa['date_d'] <= (dfa['date_d'].min() + pd.DateOffset(days=window_days)).date()]

        # Normalise the data within the rolling window
        dfar = normalise(df=dfa, model=model, feature_names=feature_names, variables_resample=variables_resample,
                         n_samples=n_samples, n_cores=n_cores, seed=seed, verbose=False)
        dfar.rename(columns={'normalised': 'rolling_' + str(i)}, inplace=True)

        # Concatenate the results
        dfr = pd.concat([dfr, dfar['rolling_' + str(i)]], axis=1)

        # Calculate ETA for the current rolling window iteration
        if verbose:
            if i==0:
                start_time = time.time()  # Initialize start time before the loop
                print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), f": Rolling window {i} from",dfa['date'].min().strftime('%Y-%m-%d'),
                    "to", dfa['date'].max().strftime('%Y-%m-%d'))

            elif i % 10 == 0:
                elapsed_time = time.time() - start_time
                remaining_time = elapsed_time / i * (len(pd.to_datetime(df['date_d'][df['date_d'] <= date_max.date()]).unique()[::rollingevery]) - i)
                if remaining_time < 60:
                    eta_str = "ETA: {:.2f} seconds".format(remaining_time)
                elif remaining_time < 3600:
                    eta_str = "ETA: {:.2f} minutes".format(remaining_time / 60)
                else:
                    eta_str = "ETA: {:.2f} hours".format(remaining_time / 3600)
                print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), f": Rolling window {i} from",dfa['date'].min().strftime('%Y-%m-%d'),
                    "to", dfa['date'].max().strftime('%Y-%m-%d'), f"{eta_str}")

    return dfr, mod_stats


def rolling_met(df=None, model=None, value=None, feature_names=None, split_method='random', fraction=0.75,
                model_config=None, n_samples=300, window_days=14, rollingevery=7, seed=7654321, n_cores=None, verbose=True):
    """
    Applies a rolling window approach to decompose the time series into different components using machine learning models.

    Parameters:
        df (pandas.DataFrame): Input dataframe containing the time series data.
        model (object, optional): Pre-trained model to use for decomposition. If None, a new model will be trained. Default is None.
        value (str): Column name of the target variable.
        feature_names (list of str): List of feature column names.
        split_method (str, optional): Method to split the data ('random' or other methods). Default is 'random'.
        fraction (float, optional): Fraction of data to be used for training. Default is 0.75.
        model_config (dict, optional): Configuration dictionary for model training parameters.
        n_samples (int, optional): Number of samples for normalisation. Default is 300.
        window_days (int, optional): Number of days for the rolling window. Default is 14.
        rollingevery (int, optional): Rolling interval in days. Default is 7.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        n_cores (int, optional): Number of cores to be used. Default is total CPU cores minus one.
        verbose (bool, optional): Whether to print progress messages. Default is True.

    Returns:
        df_dew (pandas.DataFrame): Dataframe with decomposed components including mean and standard deviation of the rolling window.
        mod_stats (pandas.DataFrame): Dataframe with model statistics.

    Example:
        >>> df = pd.read_csv('timeseries_data.csv')
        >>> value = 'target'
        >>> feature_names = ['feature1', 'feature2', 'feature3']
        >>> df_dew, mod_stats = rolling_met(df, value, feature_names, window_days=14, rollingevery=2)
    """
    if model is None:
        df, model = prepare_train_model(df, value, feature_names, split_method, fraction, model_config, seed, verbose=True)

    # Gather model statistics for testing, training, and all data
    mod_stats = pd.concat([
        modStats(df, model, set='testing'),
        modStats(df, model, set='training'),
        modStats(df.assign(set="all"), model, set='all')], ignore_index=True)

    # Variables to be used in resampling
    variables_resample = [item for item in feature_names if item not in ['hour', 'weekday', 'day_julian', 'date_unix']]

    # Default logic for cpu cores
    n_cores = n_cores if n_cores is not None else os.cpu_count() - 1

    # Initialize dataframe for decomposed components
    df_dew = pd.DataFrame(index=df.index)

    # Normalise the entire data for mean calculation
    df_dew = normalise(df, model, feature_names=feature_names, variables_resample=variables_resample, n_samples=n_samples,
                       n_cores=n_cores, seed=seed, verbose=False)

    # Initialize the dataframe for rolling window results
    dfr = pd.DataFrame(index=df_dew.index)
    df['date_d'] = pd.to_datetime(df['date']).dt.date
    date_max = pd.to_datetime(df['date_d'].max() - pd.DateOffset(days=window_days - 1))
    date_min = pd.to_datetime(df['date_d'].min() + pd.DateOffset(days=window_days - 1))

    # Apply the rolling window approach
    for i, ds in enumerate(pd.to_datetime(df['date_d'][df['date_d'] <= date_max.date()]).unique()[::rollingevery]):
        dfa = df[df['date_d'] >= ds.date()]
        dfa = dfa[dfa['date_d'] <= (dfa['date_d'].min() + pd.DateOffset(days=window_days)).date()]
        dfar = normalise(dfa, model, feature_names=feature_names, variables_resample=variables_resample, n_samples=n_samples,
                         n_cores=n_cores, seed=seed, verbose=False)
        dfar.rename(columns={'normalised': 'rolling_' + str(i)}, inplace=True)

        # Concatenate the results
        dfr = pd.concat([dfr, dfar['rolling_' + str(i)]], axis=1)

        # Calculate ETA for the current rolling window iteration
        if verbose:
            if i==0:
                start_time = time.time()  # Initialize start time before the loop
                print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), f": Rolling window {i} from",dfa['date'].min().strftime('%Y-%m-%d'), "to",
                    dfa['date'].max().strftime('%Y-%m-%d'))
            elif i % 10 == 0:
                elapsed_time = time.time() - start_time
                remaining_time = elapsed_time / i * (
                    len(pd.to_datetime(df['date_d'][df['date_d'] <= date_max.date()]).unique()[::rollingevery]) - i)
                if remaining_time < 60:
                    eta_str = "ETA: {:.2f} seconds".format(remaining_time)
                elif remaining_time < 3600:
                    eta_str = "ETA: {:.2f} minutes".format(remaining_time / 60)
                else:
                    eta_str = "ETA: {:.2f} hours".format(remaining_time / 3600)
                print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), f": Rolling window {i} from",dfa['date'].min().strftime('%Y-%m-%d'), "to",
                    dfa['date'].max().strftime('%Y-%m-%d'), f"{eta_str}")

    # Calculate the mean and standard deviation for the rolling window
    df_dew['emi_mean_' + str(window_days)] = np.mean(dfr.iloc[:, 1:], axis=1)
    df_dew['emi_std_' + str(window_days)] = np.std(dfr.iloc[:, 1:], axis=1)

    # Calculate the short-term and seasonal components
    df_dew['met_short'] = df_dew['observed'] - df_dew['emi_mean_' + str(window_days)]
    df_dew['met_season'] = df_dew['emi_mean_' + str(window_days)] - df_dew['normalised']

    return df_dew, mod_stats


def modStats(df, model, set=None, statistic=None):
    """
    Calculates statistics for model evaluation based on provided data.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing the dataset.
        model (object): Trained ML model.
        set (str, optional): Set type for which statistics are calculated ('training', 'testing', or 'all'). Default is None.
        statistic (list of str, optional): List of statistics to calculate. Default is ["n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA", "R2"].

    Returns:
        pd.DataFrame: DataFrame containing calculated statistics.

    Example:
        >>> df = pd.read_csv('timeseries_data.csv')
        >>> model = train_model(df, 'target', feature_names)
        >>> stats = modStats(df, model, set='testing')
    """
    if statistic is None:
        statistic = ["n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA","R2"]

    if set is not None:
        if 'set' in df.columns:
            df = df[df['set'] == set]
        else:
            raise ValueError(f"The DataFrame does not contain the 'set' column but 'set' parameter was provided as '{set}'.")

    df = df.assign(value_predict=model.predict(df))
    df_stats = Stats(df, mod="value_predict", obs="value", statistic=statistic).assign(set=set)

    return df_stats


def Stats(df, mod, obs,
             statistic = None):
    """
    Calculates specified statistics based on provided data.

    Parameters:
        df (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.
        statistic (list): List of statistics to calculate.

    Returns:
        DataFrame: DataFrame containing calculated statistics.
    """

    if statistic is None:
        statistic = ["n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA","R2"]
    res = {}
    if "n" in statistic:
        res["n"] = n(df, mod, obs)
    if "FAC2" in statistic:
        res["FAC2"] = FAC2(df, mod, obs)
    if "MB" in statistic:
        res["MB"] = MB(df, mod, obs)
    if "MGE" in statistic:
        res["MGE"] = MGE(df, mod, obs)
    if "NMB" in statistic:
        res["NMB"] = NMB(df, mod, obs)
    if "NMGE" in statistic:
        res["NMGE"] = NMGE(df, mod, obs)
    if "RMSE" in statistic:
        res["RMSE"] = RMSE(df, mod, obs)
    if "r" in statistic:
        res["r"] = r(df, mod, obs)[0]
        p_value = r(df, mod, obs)[1]
        if p_value >= 0.1:
            res["p_level"] = ""
        elif p_value < 0.1 and p_value >= 0.05:
            res["p_level"] = "+"
        elif p_value < 0.05 and p_value >= 0.01:
            res["p_level"] = "*"
        elif p_value < 0.01 and p_value >= 0.001:
            res["p_level"] = "**"
        else:
            res["p_level"] = "***"
    if "COE" in statistic:
        res["COE"] = COE(df, mod, obs)
    if "IOA" in statistic:
        res["IOA"] = IOA(df, mod, obs)
    if "R2" in statistic:
        res["R2"] = R2(df, mod, obs)

    results = {'n':res['n'], 'FAC2':res['FAC2'], 'MB':res['MB'], 'MGE':res['MGE'], 'NMB':res['NMB'],
               'NMGE':res['NMGE'],'RMSE':res['RMSE'], 'r':res['r'],'p_level':res['p_level'],
               'COE':res['COE'], 'IOA':res['IOA'], 'R2':res['R2']}

    results = pd.DataFrame([results])

    return results


def cpd_rupture(df, col_name='Normalised', window=12, n=5, model="l2"):
    """
    Detects change points in a time series using the ruptures package.

    Parameters:
        df (DataFrame): Input DataFrame containing the time series data.
        col_name (str, optional): Name of the column containing the time series data. Default is 'Normalised'.
        window (int, optional): Width of the sliding window. Default is 12.
        n (int, optional): Number of change points to detect. Default is 5.
        model (str, optional): Type of cost function model for the ruptures package. Default is "l2".

    Returns:
        DatetimeIndex: Datetime indices of detected change points.

    Example Usage:
        # Detect Change Points using the Ruptures Package
        change_points = cpd_rupture(df, col_name='TimeSeries', window=24, n=3, model="l1")
    """
    import ruptures as rpt
    # Convert to numpy array
    values = np.array(df[col_name])

    # Perform changepoint detection
    model = rpt.Window(width=window, model=model).fit(values)  # "l1", "rbf", "linear", "normal", "ar"
    result = model.predict(n_bkps=n)
    result = [x-1 for x in list(result)][:-1]

    # Convert changepoint indices to dates
    dates = df.iloc[result, :].index

    return dates

def pdp_all(model, df, feature_names=None, variables=None, training_only=True, n_cores=-1):
    """
    Computes partial dependence plots for all specified features.

    Parameters:
        model: AutoML model object.
        df (DataFrame): Input DataFrame containing the dataset.
        feature_names (list): List of feature names to compute partial dependence plots for.
        variables (list, optional): List of variables to compute partial dependence plots for. If None, defaults to feature_names.
        training_only (bool, optional): If True, computes partial dependence plots only for the training set. Default is True.
        n_cores (int, optional): Number of CPU cores to use for parallel computation. Default is -1 (uses all available cores).

    Returns:
        DataFrame: DataFrame containing the computed partial dependence plots for all specified features.

    Example Usage:
        # Compute Partial Dependence Plots for All Features
        df_predict = pdp_all(model, df, feature_names=['feature1', 'feature2', 'feature3'])
    """
    if variables is None:
        variables = feature_names
    if training_only:
        df = df[df["set"] == "training"]
    X_train, y_train = df[feature_names], df['value']

    results = Parallel(n_jobs=n_cores)(delayed(pdp_worker)(model, X_train, var) for var in variables)
    df_predict = pd.concat(results)
    df_predict.reset_index(drop=True, inplace=True)
    return df_predict


def pdp_worker(model, X_train, variable, training_only=True):
    """
    Worker function for computing partial dependence plots for a single feature.

    Parameters:
        model: AutoML model object.
        X_train (DataFrame): Input DataFrame containing the training data.
        variable (str): Name of the feature to compute partial dependence plot for.
        training_only (bool, optional): If True, computes partial dependence plot only for the training set. Default is True.

    Returns:
        DataFrame: DataFrame containing the computed partial dependence plot for the specified feature.
    """
    results = partial_dependence(estimator=model, X=X_train, features=variable, kind='individual')

    df_predict = pd.DataFrame({"value": results['grid_values'][0],
                               "pdp_mean": np.mean(results['individual'][0], axis=0),
                               'pdp_std': np.std(results['individual'][0], axis=0)})
    df_predict["variable"] = variable
    df_predict = df_predict[["variable", "value", "pdp_mean", "pdp_std"]]

    return df_predict


def scm_parallel(df, poll_col, date_col, code_col, control_pool, post_col, n_cores=-1):
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

    Example Usage:
        # Perform SCM in parallel for multiple treatment targets
        synthetic_all = scm_parallel(df, poll_col='Poll', date_col='Date', code_col='Code',
                                     control_pool=['A', 'B', 'C'], post_col='Post', n_cores=4)
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

    Example Usage:
        # Perform SCM for a single treatment target
        synthetic_data = scm(df, poll_col='Poll', date_col='Date', code_col='Code',
                             treat_target='T1', control_pool=['C1', 'C2'], post_col='Post')
    """
    x_pre_control = (df.loc[(df[code_col] != treat_target) & (df[code_col].isin(control_pool)) & (~df[post_col])]
                 .pivot(index=date_col, columns=code_col, values=poll_col)
                 .values)

    y_pre_treat_mean = (df
                    .loc[~df[post_col] & (df[code_col] == treat_target)]
                    .groupby(date_col)[poll_col]
                    .mean())

    param_grid = {'alpha': [i / 10 for i in range(1, 101)]}
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=5)
    grid_search.fit(x_pre_control, y_pre_treat_mean.values.reshape(-1, 1))
    best_alpha = grid_search.best_params_['alpha']
    ridge_final = Ridge(alpha=best_alpha, fit_intercept=False)
    ridge_final.fit(x_pre_control, y_pre_treat_mean.values.reshape(-1, 1))
    w = ridge_final.coef_.flatten()
    sc = (df[(df[code_col] != treat_target) & (df[code_col].isin(control_pool))]
          .pivot_table(index=date_col, columns=code_col, values=poll_col)
          .values) @ w

    data = (df
            [df[code_col] == treat_target][[date_col, code_col, poll_col]]
            .assign(synthetic=sc)).set_index(date_col)
    data['effects'] = data[poll_col] - data['synthetic']
    return data


def ml_syn(df, poll_col, date_col, code_col, treat_target, control_pool, cutoff_date, model_config):
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

    Example Usage:
        # Perform synthetic control using ML regression models
        synthetic_data = ml_syn(df, poll_col='Poll', date_col='Date', code_col='Code',
                                treat_target='T1', control_pool=['C1', 'C2'], cutoff_date='2020-01-01')
    """
    from flaml import AutoML
    automl = AutoML()
    dfp = (df[df[code_col].isin(control_pool + [treat_target])]).pivot_table(index=date_col, columns=code_col, values=poll_col)
    pre_dataset = dfp[dfp.index < cutoff_date]
    post_dataset = dfp[dfp.index >= cutoff_date]
    # Update default configuration with user-provided config
    if model_config is not None:
        default_model_config.update(model_config)

    automl.fit(dataframe=pre_dataset, label=treat_target, **default_model_config)
    pre_pred = automl.predict(pre_dataset)

    data = (df
            [df[code_col] == treat_target][[date_col, code_col, poll_col]]
            .assign(synthetic=automl.predict(dfp))).set_index(date_col)
    data['effects'] = data[poll_col] - data['synthetic']
    return data


def ml_syn_parallel(df, poll_col, date_col, code_col, control_pool, cutoff_date, training_time=60, n_cores=-1):
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

    Example Usage:
        # Perform synthetic control using ML regression models in parallel
        synthetic_all = ml_syn_parallel(df, poll_col='Poll', date_col='Date', code_col='Code',
                                        control_pool=['A', 'B', 'C'], cutoff_date='2020-01-01', training_time=120, n_cores=4)
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


## number of valid readings
def n(x, mod, obs):
    """
    Calculates the number of valid readings.

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        int: Number of valid readings.
    """
    x = x[[mod, obs]].dropna()
    res = x.shape[0]
    return res


## fraction within a factor of two
def FAC2(x, mod, obs):
    """
    Calculates the fraction of values within a factor of two.

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        float: Fraction of values within a factor of two.
    """
    x = x[[mod, obs]].dropna()
    ratio = x[mod] / x[obs]
    ratio = ratio.dropna()
    len = ratio.shape[0]
    if len > 0:
        res = ratio[(ratio >= 0.5) & (ratio <= 2)].shape[0] / len
    else:
        res = np.nan
    return res


## mean bias
def MB(x, mod, obs):
    """
    Calculates the mean bias.

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        float: Mean bias.
    """
    x = x[[mod, obs]].dropna()
    res = np.mean(x[mod] - x[obs])
    return res


## mean gross error
def MGE(x, mod, obs):
    """
    Calculates the mean gross error.

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        float: Mean gross error.
    """
    x = x[[mod, obs]].dropna()
    res = np.mean(np.abs(x[mod] - x[obs]))
    return res


## normalised mean bias
def NMB(x, mod, obs):
    """
    Calculates the normalised mean bias.

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        float: Normalised mean bias.
    """
    x = x[[mod, obs]].dropna()
    res = np.sum(x[mod] - x[obs]) / np.sum(x[obs])
    return res


## normalised mean gross error
def NMGE(x, mod, obs):
    """
    Calculates the normalised mean gross error.

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        float: Normalised mean gross error.
    """
    x = x[[mod, obs]].dropna()
    res = np.sum(np.abs(x[mod] - x[obs])) / np.sum(x[obs])
    return res


## root mean square error
def RMSE(x, mod, obs):
    """
    Calculates the root mean square error.

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        float: Root mean square error.
    """
    x = x[[mod, obs]].dropna()
    res = np.sqrt(np.mean((x[mod] - x[obs]) ** 2))
    return res


## correlation coefficient
def r(x, mod, obs):
    """
    Calculates the correlation coefficient.

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        tuple: Correlation coefficient and its p-value.
    """
    x = x[[mod, obs]].dropna()
    res = stats.pearsonr(x[mod], x[obs])
    return res


## Coefficient of Efficiency
def COE(x, mod, obs):
    """
    Calculates the Coefficient of Efficiency.

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        float: Coefficient of Efficiency.
    """
    x = x[[mod, obs]].dropna()
    res = 1 - np.sum(np.abs(x[mod] - x[obs])) / np.sum(np.abs(x[obs] - np.mean(x[obs])))
    return res


## Index of Agreement
def IOA(x, mod, obs):
    """
    Calculates the Index of Agreement.

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        float: Index of Agreement.
    """
    x = x[[mod, obs]].dropna()
    LHS = np.sum(np.abs(x[mod] - x[obs]))
    RHS = 2 * np.sum(np.abs(x[obs] - np.mean(x[obs])))
    if LHS <= RHS:
        res = 1 - LHS / RHS
    else:
        res = RHS / LHS - 1
    return res


#determination of coefficient
def R2(x, mod, obs):
    """
    Calculates the determination coefficient (R-squared).

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        float: Determination coefficient (R-squared).
    """
    x = x[[mod, obs]].dropna()
    X = sm.add_constant(x[obs])
    y=x[mod]
    model = sm.OLS(y, X).fit()
    res = model.rsquared
    return res
