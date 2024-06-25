import pandas as pd
import numpy as np
from datetime import datetime
from random import sample
from scipy import stats
from flaml import AutoML
from joblib import Parallel, delayed
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


def prepare_data(df, value, feature_names, na_rm=True, split_method='random', replace=False, fraction=0.75, seed=7654321):
    """
    Prepares the input DataFrame by performing data cleaning, imputation, and splitting.

    Parameters:
        df (DataFrame): Input DataFrame containing the dataset.
        value (str, optional): Name of the target variable. Default is 'value'.
        feature_names (list, optional): List of feature names. Default is None.
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
            .pipe(process_df, variables_col=feature_names + [value])
            .pipe(check_data, value=value)
            .pipe(impute_values, na_rm=na_rm)
            .pipe(add_date_variables, replace=replace)
            .pipe(split_into_sets, split_method=split_method, fraction=fraction, seed=seed)
            .reset_index(drop=True))

    return df


def process_df(df, variables_col):
    """
    Processes the DataFrame to ensure it contains necessary date and selected feature columns.

    This function checks if the date is present in the index or columns, selects the necessary features and
    the date column, and prepares the DataFrame for further analysis.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        variables_col (list of str): List of variable names to be included in the DataFrame.

    Returns:
        pd.DataFrame: Processed DataFrame containing the date and selected feature columns.

    Raises:
        ValueError: If no datetime information is found in index or 'date' column.

    Example:
        >>> df = pd.read_csv('data.csv')
        >>> variables_col = ['feature1', 'feature2', 'feature3']
        >>> processed_df = process_df(df, variables_col)
    """
    # Check if the date is in the index or columns
    if isinstance(df.index, pd.DatetimeIndex):
        date_in_index = True
    elif any(df.dtypes == 'datetime64[ns]'):
        date_in_index = False
    else:
        raise ValueError("No datetime information found in index or 'date' column.")

    if date_in_index:
        df = df.reset_index()

    time_column = df.select_dtypes(include='datetime64').columns.tolist()

    # Ensure there is exactly one datetime column
    if len(time_column) > 1:
        raise ValueError("More than one datetime column found.")

    # Select features and the target variable
    if variables_col:
        selected_columns = list(set(variables_col) & set(df.columns))
    else:
        selected_columns = df.columns.tolist()

    selected_columns = list(set(selected_columns).union(set(time_column)))

    df = df[selected_columns].rename(columns={time_column[0]: 'date'})

    return df


def check_data(df, value):
    """
    Validates and preprocesses the input DataFrame for subsequent analysis or modeling.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be checked.
    value : str
        The name of the target variable (column) to be used in the analysis.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing only the necessary columns, with appropriate checks and transformations applied.

    Raises:
    -------
    ValueError:
        If any of the following conditions are met:
        - The target variable (`value`) is not in the DataFrame columns.
        - There is no datetime information in either the index or the 'date' column.
        - The 'date' column is not of type datetime64.
        - The 'date' column contains missing values.

    Notes:
    ------
    - If the DataFrame's index is a DatetimeIndex, it is reset to a column named 'date'.
    - The target column (`value`) is renamed to 'value'.
    """

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
        df (DataFrame): Input DataFrame containing the dataset.
        na_rm (bool): Whether to remove missing values.

    Returns:
        DataFrame: DataFrame with imputed missing values.
    """
    # Remove missing values
    if na_rm:
        df = df.dropna(subset=['value']).reset_index(drop=True)
    # Numeric variables
    for col in df.select_dtypes(include=[np.number]).columns:
        df.fillna({col: df[col].median()}, inplace=True)

    # Character and categorical variables
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df.fillna({col: df[col].mode()[0]}, inplace=True)

    return df


def add_date_variables(df, replace):
    """
    Adds date-related variables to the DataFrame.

    Parameters:
        df (DataFrame): Input DataFrame containing the dataset.
        replace (bool): Whether to replace existing date variables.

    Returns:
        DataFrame: DataFrame with added date-related variables.
    """
    if replace:
        # Will replace if variables exist
        df['date_unix'] = df['date'].astype(np.int64) // 10**9
        df['day_julian'] = pd.DatetimeIndex(df['date']).dayofyear
        df['weekday'] = pd.DatetimeIndex(df['date']).weekday + 1
        df['weekday']=df['weekday'].astype("category")
        df['hour'] = pd.DatetimeIndex(df['date']).hour

    else:
        if 'date_unix' not in df.columns:
            df['date_unix'] = df['date'].apply(lambda x: x.timestamp())
        if 'day_julian' not in df.columns:
            df['day_julian'] = df['date'].apply(lambda x: x.timetuple().tm_yday)

        # An internal package's function
        if 'weekday' not in df.columns:
            df['weekday'] = df['date'].apply(lambda x: x.weekday() + 1)
            df['weekday']=df['weekday'].astype("category")

        if 'hour' not in df.columns:
            df['hour'] = df['date'].apply(lambda x: x.hour)

    return df


def split_into_sets(df, split_method, fraction,seed):
    """
    Splits the DataFrame into training and testing sets.

    Parameters:
        df (DataFrame): Input DataFrame containing the dataset.
        split_method (str): Method for splitting data ('random' or 'time_series').
        fraction (float): Fraction of the dataset to be used for training.
        seed (int): Seed for random operations.

    Returns:
        DataFrame: DataFrame with a 'set' column indicating the training or testing set.
    """
    # Add row number
    df = df.reset_index().rename(columns={'index': 'rowid'})
    if (split_method == 'random'):
        # Sample to get training set
        df_training = df.sample(frac=fraction, random_state=seed).reset_index(drop=True).assign(set="training")
        # Remove training set from input to get testing set
        df_testing = df[~df['rowid'].isin(df_training['rowid'])].assign(set="testing")
    if (split_method == 'time_series'):
        df_training = df.iloc[:int(fraction*df.shape[0]),:].reset_index(drop=True).assign(set="training")
        df_testing = df[~df['rowid'].isin(df_training['rowid'])].assign(set="testing")

    # Bind again
    df_split = pd.concat([df_training, df_testing], axis=0, ignore_index=True)
    df_split = df_split.sort_values(by='date').reset_index(drop=True)

    return df_split


def train_model(df, value, variables, model_config=None, seed=7654321):
    """
    Trains a machine learning model using the provided dataset and parameters.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the dataset.
        value (str): Name of the target variable.
        variables (list of str): List of feature variables.

    Keyword Parameters:
        model_config (dict, optional): Configuration dictionary for model training parameters.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.

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
        'time_budget': 60,  # Total running time in seconds
        'metric': 'r2',  # Primary metric for regression
        'estimator_list': ["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"],  # List of ML learners
        'task': 'regression',  # Task type
        'verbose': True  # Print progress messages
    }

    # Update default configuration with user-provided config
    if model_config is not None:
        default_model_config.update(model_config)

    # Initialize and train AutoML model
    model = AutoML()
    print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), ": Training AutoML...")
    model.fit(X_train=df_train[variables], y_train=df_train[value], **default_model_config, seed=seed)
    print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), ": Best model is", model.best_estimator,
          "with best model parameters of", model.best_config)

    return model


def prepare_train_model(df, value, feature_names, split_method, fraction, model_config, seed):
    """
    Prepares the data and trains a machine learning model using the specified configuration.

    This function combines data preparation and model training steps. It prepares the input DataFrame
    for training by selecting relevant columns and splitting the data, then trains a machine learning
    model using the provided configuration.

    Parameters:
    -----------
    df (pandas.DataFrame): The input DataFrame containing the data to be used for training.
    value (str): The name of the target variable (column) to be predicted.
    feature_names (list of str): A list of feature column names to be used in the training.
    split_method (str): The method to split the data ('random' or other supported methods).
    fraction (float): The fraction of data to be used for training.
    model_config (dict): The configuration dictionary for the AutoML model training.
    seed (int): The random seed for reproducibility.

    Returns:
    --------
    tuple:
        - pandas.DataFrame: The prepared DataFrame ready for model training.
        - object: The trained machine learning model.

    Example:
    --------
    >>> df = pd.read_csv('timeseries_data.csv')
    >>> value = 'target'
    >>> feature_names = ['feature1', 'feature2', 'feature3']
    >>> split_method = 'random'
    >>> fraction = 0.75
    >>> model_config = {...}
    >>> seed = 7654321
    >>> df_prepared, model = prepare_train_model(df, value, feature_names, split_method, fraction, model_config, seed)
    """

    # Prepare the data
    df = prepare_data(df, value=value, feature_names=feature_names, split_method=split_method, fraction=fraction, seed=seed)

    # Train the model using AutoML
    model = train_model(df, value='value', variables=feature_names, model_config=model_config, seed=seed)

    return df, model


def normalise_worker(index, df, model, variables_resample, replace, seed, verbose, weather_df=None):
    """
    Worker function for parallel normalization of data using randomly resampled meteorological parameters
    from another weather DataFrame within its date range. If no weather DataFrame is provided,
    it defaults to using the input DataFrame.

    Parameters:
        index (int): Index of the worker.
        df (pd.DataFrame): Input DataFrame containing the dataset.
        model (ML): Trained ML model.
        variables_resample (list of str): List of resampling variables.
        replace (bool): Whether to sample with replacement.
        seed (int): Random seed.
        verbose (bool): Whether to print progress messages.
        weather_df (pd.DataFrame, optional): Weather DataFrame containing the meteorological parameters.
                                             Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing normalized predictions.
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
    n_rows = weather_df.shape[0]
    index_rows = np.random.choice(range(n_rows), size=n_rows, replace=replace)

    # Sample meteorological parameters from weather_df
    sampled_meteorological_params = weather_df[variables_resample].iloc[index_rows].reset_index(drop=True)

    # Use the sampled parameters directly (without modifying df)
    sampled_indices = np.random.choice(sampled_meteorological_params.index, size=len(df), replace=True)
    df[variables_resample] = df[variables_resample].iloc[sampled_indices].reset_index(drop=True)

    # Predict using the model
    value_predict = model.predict(df)

    # Build DataFrame of predictions
    predictions = pd.DataFrame({
        'date': df['date'],
        'observed': df['value'],
        'normalised': value_predict
    })
    predictions['Seed'] = seed

    return predictions


def normalise(df, model, feature_names, variables_resample=None, n_samples=300, replace=True,
              aggregate=True, seed=7654321, n_cores=None, verbose=True, weather_df=None):
    """
    Normalizes the dataset using the trained model.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the dataset.
        model (object): Trained ML model.
        feature_names (list of str): List of feature names.
        variables_resample (list of str): List of resampling variables.
        n_samples (int, optional): Number of samples to normalize. Default is 300.
        replace (bool, optional): Whether to replace existing data. Default is True.
        aggregate (bool, optional): Whether to aggregate results. Default is True.
        seed (int, optional): Random seed. Default is 7654321.
        n_cores (int, optional): Number of CPU cores to use. Default is None.
        verbose (bool, optional): Whether to print progress messages. Default is True.
        weather_df (pd.DataFrame, optional): DataFrame containing weather data for resampling. Default is None.

    Returns:
        pd.DataFrame: DataFrame containing normalized predictions.

    Example:
        >>> df = pd.read_csv('timeseries_data.csv')
        >>> model = train_model(df, 'target', feature_names)
        >>> feature_names = ['feature1', 'feature2', 'feature3']
        >>> variables_resample = ['feature1', 'feature2']
        >>> normalized_df = normalise(df, model, feature_names, variables_resample)
    """
    # Default logic for cpu cores
    n_cores = n_cores if n_cores is not None else -1

    # Use all variables except the trend term
    if variables_resample is None:
        variables_resample = [var for var in feature_names if var != 'date_unix']

    # If no weather_df is provided, use df as the weather data
    if weather_df is None:
        weather_df = df

    weather_df=process_df(weather_df, variables_resample)
    # Sample the time series
    if verbose:
        print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), ": Resampling data range from", weather_df['date'].min().strftime('%Y-%m-%d'), "to", weather_df['date'].max().strftime('%Y-%m-%d'))
        print("Resampling variables:", str(variables_resample), "and predicting", n_samples, "times...")

    # If no samples are passed
    np.random.seed(seed)
    random_seeds = np.random.choice(np.arange(1000001), size=n_samples, replace=False)

    if n_samples > 0:
        # Perform normalization using parallel processing
        df_result = pd.concat(Parallel(n_jobs=n_cores)(delayed(normalise_worker)(
            index=i, df=df, model=model, variables_resample=variables_resample, replace=replace,
            seed=random_seeds[i],
            verbose=False, weather_df=weather_df) for i in range(n_samples)), axis=0)
    else:
        df_result = pd.DataFrame()

    # Aggregate results if needed
    if aggregate:
        if verbose:
            print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), ": Aggregating", n_samples, "predictions...")
        df_result = df_result[['date', 'observed', 'normalised']].pivot_table(index='date', aggfunc='mean')[['observed', 'normalised']]

    else:
        # Pivot table to reshape 'normalised' values by 'Seed' and set 'date' as index
        normalized_pivot = df_result.pivot_table(index='date', columns='Seed', values='normalised')

        # Select and drop duplicate rows based on 'date', keeping only 'observed' column
        observed_unique = df_result[['date', 'observed']].drop_duplicates().set_index('date')

        # Concatenate the pivoted 'normalised' values and unique 'observed' values
        df_result = pd.concat([observed_unique, normalized_pivot], axis=1)
        if verbose:
            print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), ": Concatenated", n_samples, "predictions...")

    return df_result


def do_all(df=None, model=None, value=None, feature_names=None, variables_resample=None, split_method='random', fraction=0.75,
           model_config=None, n_samples=300, seed=7654321, n_cores=-1, aggregate=True, weather_df=None):
    """
    Conducts data preparation, model training, and normalization, returning the transformed dataset and model statistics.

    This function performs the entire pipeline from data preparation to model training and normalization using
    specified parameters and returns the transformed dataset along with model statistics.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the dataset.
        model (object, optional): Pre-trained model to use for decomposition. If None, a new model will be trained. Default is None.
        value (str): Name of the target variable.
        feature_names (list of str): List of feature names.
        variables_resample (list of str): List of variables for normalization.
        split_method (str, optional): Method for splitting data ('random' or 'time_series'). Default is 'random'.
        fraction (float, optional): Fraction of the dataset to be used for training. Default is 0.75.
        model_config (dict, optional): Configuration dictionary for model training parameters.
        n_samples (int, optional): Number of samples for normalization. Default is 300.
        seed (int, optional): Seed for random operations. Default is 7654321.
        n_cores (int, optional): Number of CPU cores to be used for normalization (-1 for all available cores). Default is -1.
        weather_df (pd.DataFrame, optional): DataFrame containing weather data for resampling. Default is None.

    Returns:
        tuple:
            - df_dew (pd.DataFrame): Transformed dataset with normalized values.
            - mod_stats (pd.DataFrame): DataFrame containing model statistics.

    Example:
        >>> df = pd.read_csv('timeseries_data.csv')
        >>> value = 'target'
        >>> feature_names = ['feature1', 'feature2', 'feature3']
        >>> variables_resample = ['feature1', 'feature2']
        >>> df_dew, mod_stats = do_all(df, value, feature_names, variables_resample)
    """
    if model is None:
        df, model= prepare_train_model(df, value, feature_names, split_method, fraction, model_config, seed)

    # Collect model statistics
    mod_stats = pd.concat([
        modStats(df, model, set='testing'),
        modStats(df, model, set='training'),
        modStats(df.assign(set="all"), model, set='all')
    ])

    # Normalize the data using weather_df if provided
    df_dew = normalise(df, model, feature_names=feature_names, variables_resample=variables_resample, n_samples=n_samples,
                       aggregate=aggregate, n_cores=n_cores, seed=seed, verbose=True, weather_df=weather_df)

    return df_dew, mod_stats


def do_all_unc(df=None, value=None, feature_names=None, variables_resample=None, split_method='random', fraction=0.75,
               model_config=None, n_samples=300, n_models=10, confidence_level=0.95, seed=7654321, n_cores=-1, weather_df=None):
    """
    Performs uncertainty quantification by training multiple models with different random seeds and calculates statistical metrics.

    This function performs the entire pipeline from data preparation to model training and normalization, with an added step
    to quantify uncertainty by training multiple models using different random seeds. It returns a dataframe containing observed
    values, mean, standard deviation, median, confidence bounds, and weighted values, as well as a dataframe with model statistics.

    Parameters:
        df (pd.DataFrame): Input dataframe containing the time series data.
        value (str): Column name of the target variable.
        feature_names (list of str): List of feature column names.
        variables_resample (list of str): List of sampled feature names for normalization.
        split_method (str, optional): Method to split the data ('random' or other methods). Default is 'random'.
        fraction (float, optional): Fraction of data to be used for training. Default is 0.75.
        model_config (dict, optional): Configuration dictionary for model training parameters.
        n_samples (int, optional): Number of samples for normalization. Default is 300.
        n_models (int, optional): Number of models to train for uncertainty quantification. Default is 10.
        confidence_level (float, optional): Confidence level for the uncertainty bounds. Default is 0.95.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        n_cores (int, optional): Number of cores to be used (-1 for all available cores). Default is -1.
        weather_df (pd.DataFrame, optional): DataFrame containing weather data for resampling. Default is None.

    Returns:
        tuple:
            - df_dew (pd.DataFrame): Dataframe with observed values, mean, standard deviation, median, lower and upper bounds, and weighted values.
            - mod_stats (pd.DataFrame): Dataframe with model statistics.

    Example:
        >>> df = pd.read_csv('timeseries_data.csv')
        >>> value = 'target'
        >>> feature_names = ['feature1', 'feature2', 'feature3']
        >>> variables_resample = ['feature1', 'feature2']
        >>> df_dew, mod_stats = do_all_unc(df, value, feature_names, variables_resample)
    """

    np.random.seed(seed)
    random_seeds = np.random.choice(np.arange(1000001), size=n_models, replace=False)

    df_dew_list = []
    mod_stats_list = []

    for i in random_seeds:
        df_dew0, mod_stats0 = do_all(df, value=value, feature_names=feature_names,
                                     variables_resample=variables_resample,
                                     split_method=split_method, fraction=fraction,
                                     model_config=model_config,
                                     n_samples=n_samples, seed=i, n_cores=n_cores, weather_df=weather_df)

        df_dew0.rename(columns={'normalised': f'normalised_{i}'}, inplace=True)
        df_dew0 = df_dew0[['observed', f'normalised_{i}']]
        df_dew_list.append(df_dew0)

        mod_stats0['seed'] = i
        mod_stats_list.append(mod_stats0)

    df_dew = pd.concat(df_dew_list, axis=1)

    # Keep only the first 'observed' column and drop duplicates
    observed_columns = [col for col in df_dew.columns if 'observed' in col]
    df_dew = df_dew.loc[:, ~df_dew.columns.duplicated()]

    mod_stats = pd.concat(mod_stats_list, ignore_index=True)

    # Calculate statistics
    df_dew['mean'] = df_dew.iloc[:, 1:n_models + 1].mean(axis=1)
    df_dew['std'] = df_dew.iloc[:, 1:n_models + 1].std(axis=1)
    df_dew['median'] = df_dew.iloc[:, 1:n_models + 1].median(axis=1)
    df_dew['lower_bound'] = df_dew.iloc[:, 1:n_models + 1].apply(lambda x: np.quantile(x, (1 - confidence_level) / 2), axis=1)
    df_dew['upper_bound'] = df_dew.iloc[:, 1:n_models + 1].apply(lambda x: np.quantile(x, 1 - (1 - confidence_level) / 2), axis=1)

    # Calculate weighted R2
    test_stats = mod_stats[mod_stats['set'] == 'testing']
    test_stats['R2'] = test_stats['R2'].replace([np.inf, -np.inf], np.nan)
    normalized_R2 = (test_stats['R2'] - test_stats['R2'].min()) / (test_stats['R2'].max() - test_stats['R2'].min())
    weighted_R2 = normalized_R2 / normalized_R2.sum()

    # Apply weighted R2 to df_dew (excluding 'observed' column)
    df_dew_weighted = df_dew.copy()
    df_dew_weighted.iloc[:, 1:n_models + 1] = df_dew.iloc[:, 1:n_models + 1].values * weighted_R2.values
    df_dew['weighted'] = df_dew_weighted.iloc[:, 1:n_models + 1].sum(axis=1)

    return df_dew, mod_stats


def decom_emi(df=None, model=None, value=None, feature_names=None, split_method='random', fraction=0.75,
             model_config=None, n_samples=300, seed=7654321, n_cores=-1):
    """
    Decomposes a time series into different components using machine learning models.

    This function prepares the data, trains a machine learning model using AutoML, and decomposes
    the time series data into various components. The decomposition is based on the contribution
    of different features to the target variable. It returns the decomposed data and model statistics.

    Parameters:
        df (pd.DataFrame): Input dataframe containing the time series data.
        model (object, optional): Pre-trained model to use for decomposition. If None, a new model will be trained. Default is None.
        value (str): Column name of the target variable.
        feature_names (list of str): List of feature column names.
        split_method (str, optional): Method to split the data ('random' or other methods). Default is 'random'.
        fraction (float, optional): Fraction of data to be used for training. Default is 0.75.
        model_config (dict, optional): Configuration dictionary for model training parameters.
        n_samples (int, optional): Number of samples for normalization. Default is 300.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        n_cores (int, optional): Number of cores to be used (-1 for all available cores). Default is -1.

    Returns:
        tuple:
            - df_dewc (pd.DataFrame): Dataframe with decomposed components.
            - mod_stats (pd.DataFrame): Dataframe with model statistics.

    Example:
        >>> df = pd.read_csv('timeseries_data.csv')
        >>> value = 'target'
        >>> feature_names = ['feature1', 'feature2', 'feature3']
        >>> df_dewc, mod_stats = decom_emi(df, value, feature_names)
    """
    if model is None:
        df, model= prepare_train_model(df, value, feature_names, split_method, fraction, model_config, seed)

    # Gather model statistics for testing, training, and all data
    mod_stats = pd.concat([modStats(df, model, set='testing'),
                           modStats(df, model, set='training'),
                           modStats(df.assign(set="all"), model, set='all')])

    # Initialize the dataframe for decomposed components
    df_dew = df[['date', 'value']].set_index('date').rename(columns={'value': 'observed'})

    # Decompose the time series by excluding different features
    var_names = feature_names
    for var_to_exclude in ['base', 'date_unix', 'day_julian', 'weekday', 'hour']:
        var_names = list(set(var_names) - set([var_to_exclude]))
        df_dew_temp = normalise(df, model, feature_names=feature_names, variables_resample=var_names,
                                n_samples=n_samples, n_cores=n_cores, seed=seed)

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
                model_config=None, n_samples=300, seed=7654321, importance_ascending=False, n_cores=-1):
    """
    Decomposes a time series into different components using machine learning models with feature importance ranking.

    This function prepares the data, trains a machine learning model using AutoML, and decomposes the time series data
    into various components. The decomposition is based on the feature importance ranking and their contributions to the
    target variable. It returns the decomposed data and model statistics.

    Parameters:
        df (pd.DataFrame): Input dataframe containing the time series data.
        model (object, optional): Pre-trained model to use for decomposition. If None, a new model will be trained. Default is None.
        value (str): Column name of the target variable.
        feature_names (list of str): List of feature column names.
        split_method (str, optional): Method to split the data ('random' or other methods). Default is 'random'.
        fraction (float, optional): Fraction of data to be used for training. Default is 0.75.
        model_config (dict, optional): Configuration dictionary for model training parameters.
        n_samples (int, optional): Number of samples for normalization. Default is 300.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        importance_ascending (bool, optional): Sort order for feature importances. Default is False.
        n_cores (int, optional): Number of cores to be used (-1 for all available cores). Default is -1.

    Returns:
        df_dewwc (pd.DataFrame): Dataframe with decomposed components.
        mod_stats (pd.DataFrame): Dataframe with model statistics.

    Example:
        >>> df = pd.read_csv('timeseries_data.csv')
        >>> value = 'target'
        >>> feature_names = ['feature1', 'feature2', 'feature3']
        >>> df_dewwc, mod_stats = decom_met(df, value, feature_names)
    """
    if model is None:
        df, model= prepare_train_model(df, value, feature_names, split_method, fraction, model_config, seed)

    # Gather model statistics for testing, training, and all data
    mod_stats = pd.concat([
        modStats(df, model, set='testing'),
        modStats(df, model, set='training'),
        modStats(df.assign(set="all"), model, set='all')
    ])

    # Determine feature importances and sort them

    modelfi = pd.DataFrame(data={'feature_importances': model.feature_importances_},
                            index=model.feature_names_in_).sort_values('feature_importances', ascending=importance_ascending)

    # Initialize the dataframe for decomposed components
    df_deww = df[['date', 'value']].set_index('date').rename(columns={'value': 'observed'})
    met_list = ['deweathered'] + [item for item in modelfi.index if item not in ['hour', 'weekday', 'day_julian', 'date_unix']]
    var_names = [item for item in modelfi.index if item not in ['hour', 'weekday', 'day_julian', 'date_unix']]

    # Decompose the time series by excluding different features based on their importance
    for var_to_exclude in met_list:
        var_names = list(set(var_names) - set([var_to_exclude]))
        df_dew_temp = normalise(df, model, feature_names=feature_names, variables_resample=var_names, n_samples=n_samples, n_cores=n_cores, seed=seed)
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
                fraction=0.75, model_config=None, n_samples=300, window_days=14, rollingevery=2,  seed=7654321, n_cores=-1):
    """
    Applies a rolling window approach to decompose the time series into different components using machine learning models.

    This function prepares the data, trains a machine learning model using AutoML, and applies a rolling window approach
    to decompose the time series data into various components. The decomposition is based on the contribution of different
    features to the target variable over rolling windows. It returns the decomposed data and model statistics.

    Parameters:
        df (pd.DataFrame): Input dataframe containing the time series data.
        model (object, optional): Pre-trained model to use for decomposition. If None, a new model will be trained. Default is None.
        value (str): Column name of the target variable.
        feature_names (list of str): List of feature column names.
        variables_resample (list of str): List of sampled feature names for normalization.
        split_method (str, optional): Method to split the data ('random' or other methods). Default is 'random'.
        fraction (float, optional): Fraction of data to be used for training. Default is 0.75.
        model_config (dict, optional): Configuration dictionary for model training parameters.
        n_samples (int, optional): Number of samples for normalization. Default is 300.
        window_days (int, optional): Number of days for the rolling window. Default is 14.
        rollingevery (int, optional): Rolling interval in days. Default is 2.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        n_cores (int, optional): Number of cores to be used (-1 for all available cores). Default is -1.

    Returns:
        dfr (pd.DataFrame): Dataframe with rolling decomposed components.
        mod_stats (pd.DataFrame): Dataframe with model statistics.

    Example:
        >>> df = pd.read_csv('timeseries_data.csv')
        >>> value = 'target'
        >>> feature_names = ['feature1', 'feature2', 'feature3']
        >>> variables_resample = ['feature1', 'feature2']
        >>> dfr, mod_stats = rolling_dew(df, value, feature_names, variables_resample)
    """
    if model is None:
        df, model= prepare_train_model(df, value, feature_names, split_method, fraction, model_config, seed)

    # Collect model statistics
    mod_stats = pd.concat([
        modStats(df, model, set='testing'),
        modStats(df, model, set='training'),
        modStats(df.assign(set="all"), model, set='all')
    ])

    # Create an initial dataframe to store observed values
    dfr = pd.DataFrame(index=df['date'], data={'observed': list(df['value'])})
    df['date_d'] = df['date'].dt.date

    # Define the rolling window range
    date_max = df['date_d'].max() - pd.DateOffset(days=window_days - 1)
    date_min = df['date_d'].min() + pd.DateOffset(days=window_days - 1)

    # Iterate over the rolling windows
    for i, ds in enumerate(pd.to_datetime(df['date_d'][df['date_d'] <= date_max.date()]).unique()[::rollingevery]):
        dfa = df[df['date_d'] >= ds.date()]
        dfa = dfa[dfa['date_d'] <= (dfa['date_d'].min() + pd.DateOffset(days=window_days)).date()]

        # Normalize the data within the rolling window
        dfar = normalise(df=dfa, model=model, feature_names=feature_names, variables_resample=variables_resample,
                         n_samples=n_samples, n_cores=n_cores, seed=seed)
        dfar.rename(columns={'normalised':'rolling_'+str(i)},inplace=True)

        # Concatenate the results
        dfr = pd.concat([dfr, dfar['rolling_'+str(i)]], axis=1)

    return dfr, mod_stats


def rolling_met(df=None, model=None, value=None, feature_names=None, split_method='random', fraction=0.75,
                model_config=None,n_samples=300, window_days=14, rollingevery=2, seed=7654321, n_cores=-1):
    """
    Applies a rolling window approach to decompose the time series into different components using machine learning models.

    This function prepares the data, trains a machine learning model using AutoML, and applies a rolling window approach
    to decompose the time series data into various components. The decomposition is based on the contribution of different
    features to the target variable. It returns the decomposed data and model statistics.

    Parameters:
        df (pd.DataFrame): Input dataframe containing the time series data.
        model (object, optional): Pre-trained model to use for decomposition. If None, a new model will be trained. Default is None.
        value (str): Column name of the target variable.
        feature_names (list of str): List of feature column names.
        split_method (str, optional): Method to split the data ('random' or other methods). Default is 'random'.
        fraction (float, optional): Fraction of data to be used for training. Default is 0.75.
        model_config (dict, optional): Configuration dictionary for model training parameters.
        n_samples (int, optional): Number of samples for normalization. Default is 300.
        window_days (int, optional): Number of days for the rolling window. Default is 14.
        rollingevery (int, optional): Rolling interval in days. Default is 2.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        n_cores (int, optional): Number of cores to be used (-1 for all available cores). Default is -1.

    Returns:
        df_dew (pd.DataFrame): Dataframe with decomposed components including mean and standard deviation of the rolling window.
        mod_stats (pd.DataFrame): Dataframe with model statistics.

    Example:
        >>> df = pd.read_csv('timeseries_data.csv')
        >>> value = 'target'
        >>> feature_names = ['feature1', 'feature2', 'feature3']
        >>> df_dew, mod_stats = rolling_met(df, value, feature_names, window_days=14, rollingevery=2)
    """
    if model is None:
        df, model= prepare_train_model(df, value, feature_names, split_method, fraction, model_config, seed)

    # Gather model statistics for testing, training, and all data
    mod_stats = pd.concat([
        modStats(df, model, set='testing'),
        modStats(df, model, set='training'),
        modStats(df.assign(set="all"), model, set='all')
    ])

    # Variables to be used in resampling
    variables_resample = [item for item in feature_names if item not in ['hour', 'weekday', 'day_julian', 'date_unix']]

    # Normalize the data
    df_dew = normalise(df, model, feature_names=feature_names, variables_resample=variables_resample, n_samples=n_samples, n_cores=n_cores, seed=seed)

    # Initialize the dataframe for rolling window results
    dfr = pd.DataFrame(index=df_dew.index)
    df['date_d'] = pd.to_datetime(df['date']).dt.date
    date_max = pd.to_datetime(df['date_d'].max() - pd.DateOffset(days=window_days - 1))
    date_min = pd.to_datetime(df['date_d'].min() + pd.DateOffset(days=window_days - 1))

    # Apply the rolling window approach
    for i, ds in enumerate(pd.to_datetime(df['date_d'][df['date_d'] <= date_max.date()]).unique()[::rollingevery]):
        dfa = df[df['date_d'] >= ds.date()]
        dfa = dfa[dfa['date_d'] <= (dfa['date_d'].min() + pd.DateOffset(days=window_days)).date()]
        dfar = normalise(dfa, model, feature_names=feature_names, variables_resample=variables_resample, n_samples=n_samples, n_cores=n_cores, seed=seed)
        dfar.rename(columns={'normalised':'rolling_'+str(i)},inplace=True)

        # Concatenate the results
        dfr = pd.concat([dfr, dfar['rolling_'+str(i)]], axis=1)

    # Calculate the mean and standard deviation for the rolling window
    df_dew['emi_mean_' + str(window_days)] = np.mean(dfr.iloc[:, 1:], axis=1)
    df_dew['emi_std_' + str(window_days)] = np.std(dfr.iloc[:, 1:], axis=1)

    # Calculate the short-term and seasonal components
    df_dew['met_short'] = df_dew['observed'] - df_dew['emi_mean_' + str(window_days)]
    df_dew['met_season'] = df_dew['emi_mean_' + str(window_days)] - df_dew['normalised']

    return df_dew, mod_stats


def modStats(df, model, set=None, statistic=["n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA", "R2"]):
    """
    Calculates statistics for model evaluation based on provided data.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the dataset.
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
    if set is not None:
        if 'set' in df.columns:
            df = df[df['set'] == set]
        else:
            raise ValueError(f"The DataFrame does not contain the 'set' column but 'set' parameter was provided as '{set}'.")

    df.loc[:, 'value_predict'] = model.predict(df)
    df_stats = Stats(df, mod="value_predict", obs="value", statistic=statistic).assign(set=set)
    return df_stats


def Stats(df, mod, obs,
             statistic = ["n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA","R2"]):
    """
    Calculates specified statistics based on provided data.

    Parameters:
        df (DataFrame): Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.
        statistic (list): List of statistics to calculate.

    Returns:
        DataFrame: DataFrame containing calculated statistics.
    """
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


## number of valid readings
def n(x, mod, obs):
    """
    Calculates the number of valid readings.

    Parameters:
        x (DataFrame): Input DataFrame containing the dataset.
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
        x (DataFrame): Input DataFrame containing the dataset.
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
        x (DataFrame): Input DataFrame containing the dataset.
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
        x (DataFrame): Input DataFrame containing the dataset.
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
        x (DataFrame): Input DataFrame containing the dataset.
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
        x (DataFrame): Input DataFrame containing the dataset.
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
        x (DataFrame): Input DataFrame containing the dataset.
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
        x (DataFrame): Input DataFrame containing the dataset.
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
        x (DataFrame): Input DataFrame containing the dataset.
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
        x (DataFrame): Input DataFrame containing the dataset.
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
        x (DataFrame): Input DataFrame containing the dataset.
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
