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
import h2o
from h2o.automl import H2OAutoML


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
            .pipe(process_date)
            .pipe(check_data, feature_names = feature_names, value = value)
            .pipe(impute_values, na_rm = na_rm)
            .pipe(add_date_variables, replace = replace)
            .pipe(split_into_sets, split_method = split_method, fraction = fraction, seed = seed)
            .reset_index(drop = True))

    return df


def process_date(df):
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

    time_columns = df.select_dtypes(include = 'datetime64').columns

    if len(time_columns) == 0:
        raise ValueError("No datetime information found in index or columns.")
    elif len(time_columns) > 1:
        raise ValueError("More than one datetime column found.")

    df = df.rename(columns = {time_columns[0]: 'date'})

    return df


def check_data(df, feature_names, value):
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

    # Select features and the date column
    selected_columns = list(set(feature_names) & set(df.columns))
    selected_columns.extend(['date', value])
    df = df[selected_columns]

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


def nm_init_h2o(n_cores=None, min_mem_size="4G", max_mem_size="16G"):
    """
    Initialize the H2O cluster for model training.

    Parameters:
        n_cores (int, optional): Number of CPU cores to allocate for H2O. Default is all available cores minus one.
        min_mem_size (str, optional): Minimum memory size for the H2O cluster. Default is "4G".
        max_mem_size (str, optional): Maximum memory size for the H2O cluster. Default is "16G".

    Returns:
        None
    """
    # Set the number of CPU cores; use all cores minus one if not specified
    if n_cores is None:
        n_cores = os.cpu_count() - 1

    try:
        # Try to retrieve an existing H2O connection
        h2o_conn = h2o.connection()
        # Raise an error if the connection is not active
        if h2o_conn.cluster_is_up() is False:
            raise RuntimeError("H2O cluster is not up")
    except Exception as e:
        # If no existing connection is found, start a new H2O instance
        print("H2O is not running. Starting H2O...")
        h2o.init(nthreads=n_cores, min_mem_size=min_mem_size, max_mem_size=max_mem_size)
        h2o.no_progress()  # Disable progress bars for cleaner output


def h2o_train_model(df, value="value", variables=None, model_config=None, seed=7654321, n_cores=None, verbose=True, max_retries=3):
    """
    Train an AutoML model using the H2O framework.

    Parameters:
        df (pandas.DataFrame): The input data for model training.
        value (str, optional): The name of the target variable. Default is "value".
        variables (list of str): List of predictor variable names.
        model_config (dict, optional): Configuration options for the model (e.g., max models, time budget).
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        n_cores (int, optional): Number of CPU cores to allocate for training. Default is all cores minus one.
        verbose (bool, optional): Whether to print progress messages. Default is True.
        max_retries (int, optional): Maximum number of retries if model training fails. Default is 3.

    Returns:
        H2OAutoML: The best model trained by H2O AutoML.
    """
    # Check if the variables list contains duplicates
    if len(variables) != len(set(variables)):
        raise ValueError("`variables` contains duplicate elements.")

    # Ensure that all predictor variables are present in the DataFrame
    if not all(var in df.columns for var in variables):
        raise ValueError("`variables` given are not within the input DataFrame.")

    # Select training data if a 'set' column is present, otherwise use the entire DataFrame
    if 'set' in df.columns:
        df_train = df[df['set'] == 'training'][[value] + variables]
    else:
        df_train = df[[value] + variables]

    # Default configuration for model training
    default_model_config = {
        'time_budget': None,           # Time budget for AutoML training (in seconds)
        'nfolds': 5,                   # Number of cross-validation folds
        'max_models': 10,              # Maximum number of models to train
        'max_mem_size': '12g',         # Maximum memory size for H2O
        'estimator_list': ['GBM'],     # List of algorithms to use in AutoML
        'verbose': verbose             # Verbose output option
    }

    # Update the default configuration with any user-provided settings
    if model_config is not None:
        default_model_config.update(model_config)

    # Determine the number of CPU cores to use; default is all available cores minus one
    n_cores = n_cores if n_cores is not None else os.cpu_count() - 1

    def train_model():
        """
        Inner function to initialize H2O and train the AutoML model.
        """
        # Initialize the H2O cluster
        nm_init_h2o(n_cores=n_cores, max_mem_size=default_model_config['max_mem_size'])

        # Convert the pandas DataFrame to an H2OFrame for model training
        df_h2o = h2o.H2OFrame(df_train)
        response = value  # Target column name
        predictors = [col for col in df_h2o.columns if col != response]  # List of predictor columns

        # Print progress message if verbose is True
        if verbose:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Training AutoML...")

        # Train the H2O AutoML model with the specified configuration
        auto_ml = H2OAutoML(max_models=default_model_config['max_models'],
                            max_runtime_secs=default_model_config['time_budget'],
                            include_algos=default_model_config['estimator_list'],
                            seed=seed)
        auto_ml.train(x=predictors, y=response, training_frame=df_h2o)

        # Print a message showing the best model obtained
        if verbose:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Best model obtained! - {auto_ml.leaderboard[0, 'model_id']}")

        return auto_ml  # Return the AutoML object

    # Retry mechanism in case training fails
    retry_count = 0
    auto_ml = None

    # Attempt to train the model, with retries if errors occur
    while auto_ml is None and retry_count < max_retries:
        retry_count += 1
        try:
            auto_ml = train_model()  # Try to train the model
        except Exception as e:
            if verbose:
                # Print an error message and retry the training process
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Error occurred - {e}")
                print(f"Retrying... (attempt {retry_count} of {max_retries})")
            # Shut down the H2O cluster in case of failure and retry
            h2o.cluster().shutdown()
            time.sleep(5)  # Wait for a few seconds before retrying

    # If all retries fail, raise an error
    if auto_ml is None:
        raise RuntimeError(f"Failed to train the model after {max_retries} attempts.")

    # Return the best model obtained from AutoML
    return auto_ml.leader


def flaml_train_model(df, value='value', variables=None, model_config=None, seed=7654321, verbose=True):
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
        'time_budget': 90,                     # Total running time in seconds
        'metric': 'r2',                        # Primary metric for regression, 'mae', 'mse', 'r2', 'mape',...
        'estimator_list': ["lgbm"],            # List of ML learners: "lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"
        'task': 'regression',                  # Task type
        'eval_method': 'auto',                 # A string of resampling strategy, one of ['auto', 'cv', 'holdout'].
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

def train_model(df, value="value", automl_pkg="flaml", variables=None, model_config=None, seed=7654321, n_cores=None, verbose=True):
    """
    Trains a machine learning model using either FLAML or H2O AutoML.

    Parameters:
        df (pandas.DataFrame): Input dataset to train the model.
        value (str): The name of the target column in the dataset. Default is "value".
        automl_pkg (str): The AutoML package to use ("flaml" or "h2o").
        variables (list, optional): List of feature variables to use for training.
        model_config (dict, optional): Configuration settings for the model training.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        n_cores (int, optional): Number of CPU cores to use for training. Default is None.
        verbose (bool, optional): Whether to print detailed logs. Default is True.

    Returns:
        model: Trained machine learning model with a custom attribute `_model_type` indicating the package used.
    """
    if automl_pkg == "flaml":
        # Train the model using FLAML AutoML
        model = flaml_train_model(df, value, variables, model_config, seed, verbose)
        model._model_type = "flaml"  # Set custom attribute to track model type
    elif automl_pkg == "h2o":
        # Train the model using H2O AutoML
        model = h2o_train_model(df, value, variables, model_config, seed, verbose)
        model._model_type = "h2o"  # Set custom attribute to track model type
    return model


def nm_predict(model, newdata, parallel=True):
    """
    Predict using the provided trained model, either H2O or FLAML.

    Parameters:
        model: The trained machine learning model (H2O or FLAML).
        newdata (pd.DataFrame): The input data on which predictions will be made.
        parallel (bool, optional): Whether to use multi-threading during prediction (H2O models only). Default is True.

    Returns:
        np.ndarray: The predicted values from the model.
    """
    try:
        # Check the model type to determine how to perform the prediction
        if getattr(model, '_model_type', None) == 'h2o':
            # For H2O models, convert the new data to H2OFrame and make predictions
            newdata_h2o = h2o.H2OFrame(newdata)
            value_predict = model.predict(newdata_h2o).as_data_frame(use_multi_thread=parallel)['predict'].values
        elif getattr(model, '_model_type', None) == 'flaml':
            # For FLAML models, directly use the model's predict method
            value_predict = model.predict(newdata)
        else:
            # Raise an error if the model type is unsupported
            raise TypeError("Unsupported model type. The model must be an H2O or FLAML model.")

    except AttributeError as e:
        # Handle missing 'predict' method or incorrect data format
        print("Error: The model does not have a 'predict' method or 'newdata' is not in the correct format.")
        raise e
    except Exception as e:
        # Catch any other unexpected errors during prediction
        print("An unexpected error occurred during prediction.")
        raise e

    return value_predict


def generate_resampled(index, df, variables_resample, replace, seed, verbose, weather_df=None):
    """
    Worker function for parallel data normalization using randomly resampled meteorological parameters.

    Parameters:
        index (int): Index of the current row in the dataframe.
        df (pandas.DataFrame): The dataframe to be normalized.
        variables_resample (list): List of meteorological variables to be resampled.
        replace (bool): Whether to allow replacement during sampling.
        seed (int): Random seed for reproducibility.
        verbose (bool): Whether to print detailed logs during resampling.
        weather_df (pandas.DataFrame, optional): Optional dataframe of weather data to sample from.

    Returns:
        pandas.DataFrame: The dataframe with resampled meteorological parameters.
    """
    # Print progress message every 5 iterations if verbose is enabled
    if verbose and index % 5 == 0:
        message_percent = round((index / len(df)) * 100, 2)
        print(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}: Processing {index} of {len(df)} rows ({message_percent:.1f}%)...")

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Check if weather_df has the same length as df, resample meteorological data accordingly
    if len(weather_df) == len(df):
        # Sample indices with or without replacement from the dataframe
        index_rows = np.random.choice(len(df), size=len(df), replace=replace)
        df.loc[:, variables_resample] = df.loc[index_rows, variables_resample].reset_index(drop=True)
    else:
        # Sample from the weather_df if available and resample the specified variables
        sampled_params = weather_df[variables_resample].sample(n=len(df), replace=replace, random_state=seed).reset_index(drop=True)
        df.loc[:, variables_resample] = sampled_params

    return df


def normalise(df, model, feature_names, variables_resample=None, n_samples=300, replace=True,
              aggregate=True, seed=7654321, n_cores=None, weather_df=None, verbose=True):
    """
    Normalises the dataset by resampling meteorological data and using a machine learning model to make predictions.

    Parameters:
        df (pandas.DataFrame): The input DataFrame to be normalised.
        model (object): The trained machine learning model (H2O or FLAML).
        feature_names (list of str): List of feature names used for the prediction.
        variables_resample (list of str, optional): List of variables to be resampled. Default is None.
        n_samples (int, optional): Number of times to resample the data. Default is 300.
        replace (bool, optional): Whether to sample with replacement. Default is True.
        aggregate (bool, optional): Whether to aggregate the results across resamples. Default is True.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        n_cores (int, optional): Number of CPU cores to use for parallel processing. Default is all but one.
        weather_df (pandas.DataFrame, optional): Optional DataFrame containing meteorological data. Default is None (use df).
        verbose (bool, optional): Whether to print progress messages. Default is True.

    Returns:
        pandas.DataFrame: A DataFrame containing the normalised and observed values, with optional aggregation.
    """
    # Initial preprocessing on the dataset, including date processing and data validation
    df = (df.pipe(process_date)  # Process date columns
            .pipe(check_data, feature_names, 'value'))  # Check if the data contains necessary columns

    # If no external weather data is provided, use the input dataframe for weather data resampling
    if weather_df is None:
        weather_df = df

    # If variables to resample are not specified, select all features except 'date_unix'
    if variables_resample is None:
        variables_resample = [var for var in feature_names if var != 'date_unix']

    # Ensure that the weather data contains all the variables that need to be resampled
    if not all(var in weather_df.columns for var in variables_resample):
        raise ValueError("The input weather_df does not contain all variables within `variables_resample`.")

    # Prepare random seeds for parallel processing of resampling
    np.random.seed(seed)
    random_seeds = np.random.choice(np.arange(1000001), size=n_samples, replace=False)  # Generate unique random seeds

    # Determine the number of CPU cores to use for parallel processing
    n_cores = n_cores if n_cores is not None else os.cpu_count() - 1

    if verbose:
        print(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}: Normalising the dataset in parallel.")

    # Perform resampling in parallel across n_cores
    with Parallel(n_jobs=n_cores, backend='threading') as parallel:
        df_resampled_list = parallel(
            delayed(generate_resampled)(
                index=i, df=df.copy().reset_index(drop=True),  # Copy the dataframe for each sample to avoid modification issues
                variables_resample=variables_resample, replace=replace,  # Resampling parameters
                seed=random_seeds[i], verbose=False, weather_df=weather_df  # Use unique seed and weather data
            ) for i in range(n_samples)  # Repeat for the number of samples
        )

    # If verbose, print the progress of making predictions
    if verbose:
        print(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}: Predicting using trained model in batches.")

    # Concatenate all resampled DataFrames into a single DataFrame
    df_all_resampled = pd.concat(df_resampled_list, ignore_index=True)

    # Use the trained model to make predictions on the resampled data
    predictions = nm_predict(model, df_all_resampled)

    # Create a DataFrame to store the predictions and the corresponding observed values
    df_result = pd.DataFrame({
        'date': pd.concat([df_resampled['date'] for df_resampled in df_resampled_list], ignore_index=True),
        'observed': pd.concat([df_resampled['value'] for df_resampled in df_resampled_list], ignore_index=True),
        'normalised': predictions,  # Predicted values from the model
        'seed': np.repeat(random_seeds, len(df))  # Add the random seed used for each sample
    })

    # If aggregation is enabled, average the predictions across resamples for each date
    if aggregate:
        if verbose:
            print(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}: Aggregating {n_samples} predictions...")

        # Pivot the table to calculate the mean observed and normalised values for each date
        df_result = df_result.pivot_table(index='date', aggfunc='mean')[['observed', 'normalised']]

    return df_result  # Return the final normalised DataFrame


def prepare_train_model(df, value, automl_pkg, feature_names, split_method, fraction, model_config, seed, verbose=True):
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
        >>> model_config = {'time_budget': 90, 'metric': 'r2'}
        >>> seed = 7654321
        >>> df_prepared, model = prepare_train_model(df, value='target', feature_names=feature_names, split_method=split_method, fraction=fraction, model_config=model_config, seed=seed, verbose=True)
    """

    vars = list(set(feature_names) - set(['date_unix', 'day_julian', 'weekday', 'hour']))

    # Prepare the data
    df = prepare_data(df, value=value, feature_names=vars, split_method=split_method, fraction=fraction, seed=seed)

    # Train the model using AutoML
    model = train_model(df, value='value', automl_pkg= automl_pkg, variables=feature_names, model_config=model_config, seed=seed, verbose=verbose)

    return df, model

def do_all(df=None, model=None, value=None, automl_pkg='flaml', feature_names=None, variables_resample=None, split_method='random', fraction=0.75,
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
        df, model= prepare_train_model(df, value, automl_pkg, feature_names, split_method, fraction, model_config, seed, verbose)

    # Collect model statistics
    mod_stats = modStats(df, model)

    # Default logic for cpu cores
    n_cores = n_cores if n_cores is not None else os.cpu_count() - 1

    # Normalise the data using weather_df if provided
    df_dew = normalise(df, model, feature_names=feature_names, variables_resample=variables_resample, n_samples=n_samples,
                       aggregate=aggregate, n_cores=n_cores, seed=seed, weather_df=weather_df, verbose=verbose)

    return df_dew, mod_stats


def do_all_unc(df=None, value=None, automl_pkg='flaml', feature_names=None, variables_resample=None, split_method='random',
               fraction=0.75, model_config=None, n_samples=300, n_models=10, confidence_level=0.95, seed=7654321,
               n_cores=None, weather_df=None, verbose=True):
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
        df_dew0, mod_stats0 = do_all(df, value=value, automl_pkg=automl_pkg, feature_names=feature_names,
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


def decom_emi(df=None, model=None, value=None, automl_pkg='flaml', feature_names=None, split_method='random', fraction=0.75,
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
        df, model = prepare_train_model(df, value, automl_pkg, feature_names, split_method, fraction, model_config, seed, verbose=True)

    # Gather model statistics for testing, training, and all data
    mod_stats = modStats(df, model)

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


def decom_met(df=None, model=None, value=None, automl_pkg='flaml', feature_names=None, split_method='random', fraction=0.75,
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
        df, model = prepare_train_model(df, value, automl_pkg, feature_names, split_method, fraction, model_config, seed, verbose=True)

    # Gather model statistics for testing, training, and all data
    mod_stats = modStats(df, model)

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


def rolling(df=None, model=None, value=None, automl_pkg='flaml', feature_names=None, variables_resample=None, split_method='random', fraction=0.75,
            model_config=None, n_samples=300, window_days=14, rolling_every=7, seed=7654321, n_cores=None, verbose=True):
    """
    Applies a rolling window approach to decompose the time series into different components using machine learning models.

    Parameters:
        df (pandas.DataFrame): Input dataframe containing the time series data.
        model (object, optional): Pre-trained model to use for decomposition. If None, a new model will be trained. Default is None.
        value (str): Column name of the target variable.
        feature_names (list of str): List of feature column names used as features for the model.
        split_method (str, optional): Method to split the data ('random' or other methods). Default is 'random'.
        fraction (float, optional): Fraction of data to be used for training. Default is 0.75.
        model_config (dict, optional): Configuration dictionary for model training parameters.
        n_samples (int, optional): Number of samples for normalisation. Default is 300.
        window_days (int, optional): Number of days for the rolling window. Default is 14.
        rolling_every (int, optional): Rolling interval in days. Default is 7.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        n_cores (int, optional): Number of cores to be used for parallel processing. Default is total CPU cores minus one.
        verbose (bool, optional): Whether to print progress messages. Default is True.

    Returns:
        combined_results (pandas.DataFrame): Dataframe containing observed and rolling normalized results.
        mod_stats (pandas.DataFrame): Dataframe with model statistics.

    Example:
        >>> df = pd.read_csv('timeseries_data.csv')
        >>> value = 'target'
        >>> feature_names = ['feature1', 'feature2', 'feature3']
        >>> df_dew, mod_stats = rolling(df, value, feature_names, window_days=14, rolling_every=7)
    """

    # If no model is provided, train a new model on the dataset
    if model is None:
        df, model = prepare_train_model(df, value, automl_pkg, feature_names, split_method, fraction, model_config, seed, verbose=True)

    # Gather model statistics for the training, testing, and overall dataset
    mod_stats = modStats(df, model)

    # Default logic to determine the number of CPU cores for parallel computation
    n_cores = n_cores if n_cores is not None else os.cpu_count() - 1

    # Convert the date column to datetime (daily precision)
    df['date_d'] = df['date'].dt.date

    # Define the maximum and minimum date range for the rolling window
    date_max = pd.to_datetime(df['date_d'].max() - pd.DateOffset(days=window_days - 1))
    date_min = pd.to_datetime(df['date_d'].min() + pd.DateOffset(days=window_days - 1))

    # Create a list of dates for each rolling window based on the interval (rolling_every)
    rolling_dates = pd.to_datetime(df['date_d'][df['date_d'] <= date_max.date()]).unique()[::rolling_every-1]

    # Initialize a DataFrame to store the results of each rolling window
    combined_results = df.set_index('date')[['value']].rename(columns={'value': 'observed'})

    # Apply the rolling window approach
    for i, ds in enumerate(rolling_dates):
        # Filter the data for the current rolling window (from start date `ds` to window_days length)
        dfa = df[df['date_d'] >= ds.date()]
        dfa = dfa[dfa['date_d'] < (dfa['date_d'].min() + pd.DateOffset(days=window_days)).date()]

        try:
            # Normalize the data for the current rolling window
            dfar = normalise(dfa, model, feature_names=feature_names, variables_resample=variables_resample,
                             n_samples=n_samples, n_cores=n_cores, seed=seed, verbose=False)

            # Rename the 'normalised' column to indicate the rolling window index
            dfar.rename(columns={'normalised': 'rolling_' + str(i)}, inplace=True)

            # Merge the normalized results of the current window with the overall result DataFrame
            combined_results = pd.concat([combined_results, dfar['rolling_' + str(i)]], axis=1)

            # If verbose is enabled, print progress every 10 rolling windows
            if verbose and (i % 10 == 0):
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Rolling window {i} from {dfa['date'].min().strftime('%Y-%m-%d')} to {(dfa['date'].max()).strftime('%Y-%m-%d')}")

        except Exception as e:
            # Handle any errors during the normalization process and continue with the next rolling window
            if verbose:
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Error during normalization for rolling window {i} from {dfa['date'].min().strftime('%Y-%m-%d')} to {(dfa['date'].max()).strftime('%Y-%m-%d')}: {str(e)}")

    # Return the final DataFrame containing observed and rolling normalized values, along with model statistics
    return combined_results, mod_stats


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
        statistic = ["n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA", "R2"]

    def calculate_stats(df, set_name=None):
        if set_name is not None:
            if 'set' in df.columns:
                df = df[df['set'] == set_name]
            else:
                raise ValueError(f"The DataFrame does not contain the 'set' column but 'set' parameter was provided as '{set_name}'.")

        df = df.assign(value_predict=nm_predict(model,df))
        df_stats = Stats(df, mod="value_predict", obs="value", statistic=statistic).assign(set=set_name)
        return df_stats

    if set is None:
        if 'set' in df.columns:
            sets = df['set'].unique()
            stats_list = [calculate_stats(df, s) for s in sets]
            # Add statistics for the whole dataset with 'set' as "all"
            df_all = df.copy()
            df_all['set'] = 'all'
            stats_list.append(calculate_stats(df_all, 'all'))
            df_stats = pd.concat(stats_list, ignore_index=True)
        else:
            raise ValueError("The DataFrame does not contain the 'set' column and 'set' parameter was not provided.")
    else:
        df_stats = calculate_stats(df, set)

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


def extract_feature_names(model):
    """
    Extract feature names from the best estimator of an AutoML model, either 'h2o' or 'flaml'.

    Parameters:
        model (AutoML): The trained AutoML model object (either from H2O or FLAML framework).

    Returns:
        list: List of feature names used in the model.

    Raises:
        AttributeError: If the model doesn't have identifiable feature names.
        TypeError: If the model type is unsupported (not 'h2o' or 'flaml').

    Example Usage:
        # Extract feature names from a trained AutoML model
        feature_names = extract_feature_names(trained_model)
    """

    # If the model is from H2O, extract feature names using the varimp function
    if getattr(model, '_model_type', None) == 'h2o':
        feature_names = model.varimp(use_pandas=True)['variable']

    # If the model is from FLAML, check for feature names in the model attributes
    elif getattr(model, '_model_type', None) == 'flaml':
        # Check if the FLAML model has 'feature_name_' attribute and extract feature names
        if hasattr(model, 'feature_name_'):
            feature_names = model.feature_name_
        # If 'feature_name_' is not available, check for 'feature_names_in_' attribute
        elif hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        # Raise an error if no identifiable feature names are found in the FLAML model
        else:
            raise AttributeError("The best estimator does not have identifiable feature names.")

    # Raise an error if the model type is neither 'h2o' nor 'flaml'
    else:
        raise TypeError("Unsupported model type. The model must be an H2O or FLAML model.")

    # Return the feature names as a list
    return list(feature_names)


def pdp(df, model, var_list=None, training_only=True, n_cores=None):
    """
    Computes partial dependence plots (PDP) for the specified features.

    Parameters:
        df (DataFrame): The input dataset containing features and labels.
        model: Trained AutoML model object (either 'flaml' or 'h2o').
        var_list (list, optional): List of variables to compute partial dependence plots for. If None, all feature names will be used.
        training_only (bool, optional): If True, computes PDP only for the training set. Default is True.
        n_cores (int, optional): Number of CPU cores to use for parallel computation. Default is all available cores minus one.

    Returns:
        DataFrame: A DataFrame containing partial dependence plot (PDP) values for the specified features.

    Example Usage:
        # Compute Partial Dependence Plots for specified features
        df_predict = pdp(df, model, var_list=['feature1', 'feature2'])
    """

    # Extract the names of features used in the model
    feature_names = extract_feature_names(model)

    # Set the number of CPU cores for parallel processing. Default is all available cores minus one.
    n_cores = n_cores if n_cores is not None else os.cpu_count() - 1

    # If no variables are provided, use all feature names
    if var_list is None:
        var_list = feature_names

    # If `training_only` is True, filter the dataset to only include the training set
    if training_only:
        df = df[df["set"] == "training"]

    # Check if the model is of type 'h2o'
    if getattr(model, '_model_type', None) == 'h2o':
        # Convert the DataFrame into an H2OFrame for h2o models
        df = h2o.H2OFrame(df)

        # Define a helper function for computing PDP for h2o models
        def h2opdp(df, model, var):
            # Get partial dependence plot values from the h2o model
            pdp_value = model.partial_plot(frame=df, cols=[var], plot=False)[0].as_data_frame()
            # Rename the column for better readability
            pdp_value.rename(columns={var: 'value'}, inplace=True)
            # Add a new column indicating the variable for which the PDP was computed
            pdp_value["variable"] = [var] * len(pdp_value)
            return pdp_value

        # Compute PDP for each variable in the list and concatenate the results
        pdp_value_all = pd.concat([h2opdp(df, model, var) for var in var_list])

    # If the model is of type 'flaml'
    elif getattr(model, '_model_type', None) == 'flaml':
        # Prepare the feature matrix (X_train) and target variable (y_train)
        X_train, y_train = df[feature_names], df['value']

        # Use parallel processing to compute PDP for each variable in the list
        results = Parallel(n_jobs=n_cores)(delayed(pdp_worker)(X_train, model, var) for var in var_list)

        # Concatenate the results and reset the index
        pdp_value_all = pd.concat(results)
        pdp_value_all.reset_index(drop=True, inplace=True)

    # Return the final DataFrame containing PDP values for all specified features
    return pdp_value_all


def pdp_worker(X_train, model, variable, training_only=True):
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


def scm_all(df, poll_col, code_col, control_pool, cutoff_date, n_cores=None):
    """
    Performs Synthetic Control Method (SCM) in parallel for multiple treatment targets.

    Parameters:
        df (DataFrame): Input DataFrame containing the dataset.
        poll_col (str): Name of the column containing the poll data.
        code_col (str): Name of the column containing the code data.
        control_pool (list): List of control pool codes.
        cutoff_date (str): Date for splitting pre- and post-treatment datasets.
        n_cores (int, optional): Number of CPU cores to use. Default is total CPU cores minus one.

    Returns:
        DataFrame: DataFrame containing synthetic control results for all treatment targets.

    Example Usage:
        # Perform SCM in parallel for multiple treatment targets
        synthetic_all = scm_all(df, poll_col='Poll', code_col='Code',
                                     control_pool=['A', 'B', 'C'], cutoff_date='2020-01-01', n_cores=4)
    """
    # Default logic for cpu cores
    n_cores = n_cores if n_cores is not None else os.cpu_count() - 1
    treatment_pool = df[code_col].unique()
    synthetic_all = pd.concat(Parallel(n_jobs=n_cores)(delayed(scm)(
                    df=df,
                    poll_col=poll_col,
                    code_col=code_col,
                    treat_target=code,
                    control_pool=control_pool,
                    cutoff_date=cutoff_date) for code in treatment_pool))
    return synthetic_all


def scm(df, poll_col, code_col, treat_target, control_pool, cutoff_date):
    """
    Performs Synthetic Control Method (SCM) for a single treatment target.

    Parameters:
        df (DataFrame): Input DataFrame containing the dataset.
        poll_col (str): Name of the column containing the poll data.
        code_col (str): Name of the column containing the code data.
        treat_target (str): Code of the treatment target.
        control_pool (list): List of control pool codes.
        cutoff_date (str): Date for splitting pre- and post-treatment datasets.

    Returns:
        DataFrame: DataFrame containing synthetic control results for the specified treatment target.

    Example Usage:
        # Perform SCM for a single treatment target
        synthetic_data = scm(df, poll_col='Poll', code_col='Code',
                             treat_target='T1', control_pool=['C1', 'C2'], cutoff_date='2020-01-01')
    """
    df = process_date(df)

    # Splitting the dataset into pre- and post-treatment periods
    pre_treatment_df = df[df['date'] < cutoff_date]
    post_treatment_df = df[df['date'] >= cutoff_date]

    # Preparing pre-treatment control data
    x_pre_control = (pre_treatment_df.loc[(pre_treatment_df[code_col] != treat_target) &
                                          (pre_treatment_df[code_col].isin(control_pool))]
                     .pivot(index='date', columns=code_col, values=poll_col)
                     .values)

    # Preparing pre-treatment data for the treatment target
    y_pre_treat_mean = (pre_treatment_df
                        .loc[(pre_treatment_df[code_col] == treat_target)]
                        .groupby('date')[poll_col]
                        .mean())

    # Grid search to find the best alpha parameter for Ridge regression
    param_grid = {'alpha': [i / 10 for i in range(1, 101)]}
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=5)
    grid_search.fit(x_pre_control, y_pre_treat_mean.values.reshape(-1, 1))
    best_alpha = grid_search.best_params_['alpha']

    # Final Ridge regression model with the best alpha parameter, including intercept
    ridge_final = Ridge(alpha=best_alpha, fit_intercept=True)
    ridge_final.fit(x_pre_control, y_pre_treat_mean.values.reshape(-1, 1))
    w = ridge_final.coef_.flatten()
    intercept = ridge_final.intercept_.item()

    # Preparing control data for synthetic control calculation
    sc = (df[(df[code_col] != treat_target) & (df[code_col].isin(control_pool))]
          .pivot_table(index='date', columns=code_col, values=poll_col)
          .values) @ w + intercept

    # Combining synthetic control results with actual data
    data = (df[df[code_col] == treat_target][['date', code_col, poll_col]]
            .assign(synthetic=sc)).set_index('date')
    data['effects'] = data[poll_col] - data['synthetic']

    return data

def mlsc(df, poll_col, code_col, treat_target, control_pool, cutoff_date, automl_pkg='flaml', model_config=None,
         split_method='random', fraction=1, seed=7654321):
    """
    Performs a synthetic control analysis using machine learning models to estimate the treatment effect.

    Parameters:
        df (DataFrame): The input dataset containing date, treatment, and control information.
        poll_col (str): The column name of the target variable (e.g., poll results, sales).
        code_col (str): The column containing the codes for treatment and control units.
        treat_target (str): The code for the treatment target (the treated unit).
        control_pool (list): A list of codes representing the control units.
        cutoff_date (str): The date that separates pre-treatment and post-treatment periods.
        automl_pkg (str, optional): The AutoML package to use for training the model. Default is 'flaml'.
        model_config (dict, optional): Configuration for the machine learning model.
        split_method (str, optional): How to split the training data (default is 'random').
        fraction (float, optional): Fraction of the data to use for training (default is 1, meaning use all data).

    Returns:
        DataFrame: A DataFrame containing synthetic control values and estimated treatment effects.
    """

    # Preprocess the DataFrame to ensure proper formatting of date columns
    df = process_date(df)

    # Combine the control pool with the treatment target to create a union of the units to analyze
    union_result = list(set(control_pool) | set([treat_target]))

    # Create a pivot table with 'date' as the index, and control and treatment codes as columns, using poll_col as values
    dfp = df[df[code_col].isin(union_result)].pivot_table(index='date', columns=code_col, values=poll_col)

    # Split the data into pre-treatment and post-treatment datasets based on the cutoff date
    pre_dataset = dfp[dfp.index < cutoff_date]
    post_dataset = dfp[dfp.index >= cutoff_date]

    # Remove the treatment target from the control pool to avoid overlap
    control_pool_u = list(set(control_pool) - set([treat_target]))

    # Prepare the model by training it on pre-treatment data. 'dfx' contains the feature set, and 'model' is the trained model
    dfx, model = prepare_train_model(df=pre_dataset, value=treat_target, automl_pkg=automl_pkg,
                                     feature_names=control_pool_u, split_method=split_method,
                                     fraction=fraction, model_config=model_config, seed=seed)

    # Use the trained model to predict the synthetic control values for the treated unit
    data = (df[df[code_col] == treat_target][['date', code_col, poll_col]]
            .assign(synthetic=nm_predict(model, dfp)))  # Assign the predicted synthetic values
    data = data.set_index('date')  # Set the 'date' column as the index

    # Calculate the treatment effects by subtracting the synthetic values from the actual observed values
    data['effects'] = data[poll_col] - data['synthetic']

    # Return the final DataFrame containing the synthetic control values and estimated treatment effects
    return data


def mlsc_all(df, poll_col, code_col, control_pool, cutoff_date, automl_pkg='flaml', model_config=None,
         split_method='random', fraction=1, n_cores=None):
    """
    Performs synthetic control using machine learning regression models in parallel for multiple treatment targets.

    Parameters:
        df (DataFrame): Input DataFrame containing the dataset.
        poll_col (str): Name of the column containing the poll data.
        code_col (str): Name of the column containing the code data.
        control_pool (list): List of control pool codes.
        cutoff_date (str): Date for splitting pre- and post-treatment datasets.
        automl_pkg (str, optional): AutoML package to use ('flaml' or 'h2o'). Default is 'flaml'.
        model_config (dict, optional): Model configuration for the AutoML package.
        split_method (str, optional): Method for splitting training data. Default is 'random'.
        fraction (float, optional): Fraction of the data to use for training. Default is 1.
        n_cores (int, optional): Number of CPU cores to use for parallel computation. Default is all cores minus one.

    Returns:
        DataFrame: A DataFrame containing synthetic control results for all treatment targets.

    Example Usage:
        synthetic_all = mlsc_all(df, poll_col='Poll', code_col='Code',
                                 control_pool=['A', 'B', 'C'], cutoff_date='2020-01-01', n_cores=4)
    """

    # Set the number of CPU cores to use for parallel processing. If not provided, use all available cores minus one.
    n_cores = n_cores if n_cores is not None else os.cpu_count() - 1

    # Get the unique treatment targets from the dataset (all unique values in the code_col).
    treatment_pool = df[code_col].unique()

    # If the selected AutoML package is 'flaml', use a sequential processing approach.
    if automl_pkg == 'flaml':
        # For each treatment target, apply the mlsc function, and concatenate the results into one DataFrame.
        synthetic_all = pd.concat(
            [mlsc(
                df=df,
                poll_col=poll_col,
                code_col=code_col,
                treat_target=code,
                control_pool=control_pool,
                cutoff_date=cutoff_date,
                automl_pkg=automl_pkg,
                model_config=model_config,
                split_method=split_method,
                fraction=fraction
            ) for code in treatment_pool]
        )

    # If the selected AutoML package is 'h2o', use parallel processing to speed up the process.
    elif automl_pkg == 'h2o':
        # Use Parallel and delayed from joblib to apply the mlsc function to each treatment target in parallel.
        synthetic_all = pd.concat(
            Parallel(n_jobs=n_cores)(delayed(mlsc)(
                df=df,
                poll_col=poll_col,
                code_col=code_col,
                treat_target=code,
                control_pool=control_pool,
                cutoff_date=cutoff_date,
                automl_pkg=automl_pkg,
                model_config=model_config,
                split_method=split_method,
                fraction=fraction
            ) for code in treatment_pool)
        )

    # Return the combined DataFrame with synthetic control results for all treatment targets.
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
