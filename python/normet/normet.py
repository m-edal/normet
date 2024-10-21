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
import pickle

def prepare_data(df, value, feature_names, na_rm=True, split_method='random', fraction=0.75, seed=7654321):
    """
    Prepares the input DataFrame by performing data cleaning, imputation, and splitting.

    Parameters:
        df (pandas.DataFrame):: Input DataFrame containing the dataset.
        value (str): Name of the target variable.
        feature_names (list): List of feature names.
        na_rm (bool, optional): Whether to remove missing values. Default is True.
        split_method (str, optional): Method for splitting data ('random' or 'time_series'). Default is 'random'.
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
            .pipe(add_date_variables)
            .pipe(split_into_sets, split_method = split_method, fraction = fraction, seed = seed)
            .reset_index(drop = True))

    return df


def process_date(df):
    """
    Processes the DataFrame to ensure it contains necessary date and selected feature columns.

    This function checks if the date is present in the index or columns, selects the necessary features and
    the date column, and prepares the DataFrame for further analysis.

    Parameters:
        df (pandas.DataFrame): Input DataFrame that needs processing.

    Returns:
        pd.DataFrame: Processed DataFrame containing the 'date' column and other selected feature columns.

    Raises:
        ValueError: If no datetime information is found in index or columns, or if more than one datetime column is present.
    """

    # Check if the DataFrame index is a DatetimeIndex. If so, reset the index to convert the index to a column.
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # Select columns in the DataFrame that are of the datetime64 data type.
    time_columns = df.select_dtypes(include='datetime64').columns

    # If no datetime columns are found, raise a ValueError.
    if len(time_columns) == 0:
        raise ValueError("No datetime information found in index or columns.")

    # If more than one datetime column is found, raise a ValueError.
    elif len(time_columns) > 1:
        raise ValueError("More than one datetime column found.")

    # Rename the found datetime column to 'date' for consistency.
    df = df.rename(columns={time_columns[0]: 'date'})

    # Return the processed DataFrame with the 'date' column.
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


def add_date_variables(df):
    """
    Adds date-related variables to the DataFrame.

    Parameters:
        df (pandas.DataFrame):: Input DataFrame containing the dataset.

    Returns:
        DataFrame: DataFrame with added date-related variables.
    """
    df.loc[:,'date_unix'] = df['date'].astype(np.int64) // 10**9
    df.loc[:,'day_julian'] = pd.DatetimeIndex(df['date']).dayofyear
    df.loc[:,'weekday'] = pd.DatetimeIndex(df['date']).weekday + 1
    df.loc[:,'weekday'] = df['weekday'].astype("category")
    df.loc[:,'hour'] = pd.DatetimeIndex(df['date']).hour

    return df


def split_into_sets(df, split_method, fraction, seed):
    """
    Splits the DataFrame into training and testing sets.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing the dataset.
        split_method (str): Method for splitting data ('random', 'ts', 'season', 'month').
        fraction (float): Fraction of the dataset to be used for training (for 'random', 'ts', 'season')
                          or fraction of each month to be used for training (for 'month').
        seed (int): Seed for random operations.

    Returns:
        DataFrame: DataFrame with a 'set' column indicating the training or testing set.
    """
    if split_method == 'random':
        # Randomly sample for training and use the rest for testing
        df['set'] = 'testing'
        df.loc[df.sample(frac=fraction, random_state=seed).index, 'set'] = 'training'

    elif split_method == 'ts':
        # Split based on fraction for time series
        split_index = int(fraction * len(df))
        df['set'] = ['training' if i < split_index else 'testing' for i in range(len(df))]

    elif split_method == 'season':
        # Map month to season
        season_map = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
                      6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
        df['season'] = df['date'].dt.month.map(season_map)

        # Split each season
        df['set'] = 'testing'
        for season in df['season'].unique():
            season_df = df[df['season'] == season]
            split_index = int(fraction * len(season_df))
            df.loc[season_df.index[:split_index], 'set'] = 'training'

    elif split_method == 'month':
        # Split each month
        df['month'] = df['date'].dt.month
        df['set'] = 'testing'
        for month in range(1, 13):
            month_df = df[df['month'] == month]
            split_index = int(fraction * len(month_df))
            df.loc[month_df.index[:split_index], 'set'] = 'training'

    # Return the DataFrame with 'set' column, sorted by date
    return df.sort_values(by='date').reset_index(drop=True)


def init_h2o(n_cores=None, max_mem_size="16G"):
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
    n_cores = n_cores or os.cpu_count() - 1

    try:
        # Try to retrieve an existing H2O connection
        h2o_conn = h2o.connection()
        # Raise an error if the connection is not active
        if h2o_conn.cluster_is_up() is False:
            raise RuntimeError("H2O cluster is not up")
    except Exception as e:
        # If no existing connection is found, start a new H2O instance
        print("H2O is not running. Starting H2O...")
        h2o.init(nthreads=n_cores, max_mem_size=max_mem_size)
        h2o.no_progress()  # Disable progress bars for cleaner output


def save_h2o(model, path, filename):
    """
    Save an H2O model to the specified directory with the given filename.

    Parameters:
        model (AutoML): The H2O model to save.
        path (str): Directory path to save the model.
        filename (str): Desired filename for the saved model.

    Returns:
        str: The path of the saved model.
    """
    # Ensure the output directory exists
    os.makedirs(path, exist_ok=True)

    # Save the model to the specified directory
    model_path = h2o.save_model(model=model, path=path, force=True)

    # Construct the new file path with the specified filename
    new_model_path = os.path.join(path, filename)

    # Rename the model file to the desired filename
    os.rename(model_path, new_model_path)

    return new_model_path


def save_flaml(model, path, filename):
    """
    Save a FLAML AutoML model to the specified directory with the given filename.

    Parameters:
        model (AutoML): The FLAML AutoML model to save.
        path (str): Directory path to save the model.
        filename (str): Desired filename for the saved model.

    Returns:
        str: The path of the saved model.
    """
    # Ensure the output directory exists
    os.makedirs(path, exist_ok=True)

    # Construct the full path for the model file
    model_path = os.path.join(path, filename)

    # Save the FLAML model using pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

    return model_path


def load_model(automl_pkg="flaml", path='./', model_name='automl'):
    """
    Loads a pre-trained AutoML model from disk, either from the FLAML or H2O framework.

    This function loads a saved model from the specified path, either using the FLAML or H2O framework,
    and assigns a custom attribute (`_model_type`) to indicate the model's type.

    Parameters:
        automl_pkg (str, optional): Specifies the AutoML package ('flaml' or 'h2o'). Default is 'flaml'.
        path (str, optional): The directory path where the model is saved. Default is './'.
        model_name (str, optional): The name of the model file (without extension). Default is 'automl'.

    Returns:
        object: The loaded AutoML model with the `_model_type` attribute set to 'flaml' or 'h2o'.

    Raises:
        FileNotFoundError: If the model file does not exist in the specified path.
        Exception: If the model loading process fails for any other reason.

    Example:
        >>> model = load_model(automl_pkg="flaml", path="/models", model_name="my_model")
    """
    if automl_pkg == "flaml":
        # Load the FLAML model from the specified path and mark it as a FLAML model
        model_path = os.path.join(path, model_name)
        model = pickle.load(open(model_path, 'rb'))

    elif automl_pkg == "h2o":
        # Load the H2O model from the specified path and mark it as an H2O model
        model_path = os.path.join(path, model_name)
        model = h2o.load_model(model_path)

    model._model_type = automl_pkg

    return model


def _convert_h2oframe_to_numeric(h2o_frame, training_columns):
    """
    Converts the specified columns of an H2O Frame to numeric types.

    This function ensures that all the columns used for training in an H2O Frame are of numeric type,
    which is required for certain machine learning models.

    Parameters:
        h2o_frame (h2o.H2OFrame): The input H2O Frame containing the data to be converted.
        training_columns (list of str): List of column names to be converted to numeric types.

    Returns:
        h2o.H2OFrame: The modified H2O Frame with specified columns converted to numeric types.

    Example:
        >>> h2o_frame = h2o.H2OFrame(data)
        >>> training_columns = ['feature1', 'feature2']
        >>> h2o_frame = _convert_h2oframe_to_numeric(h2o_frame, training_columns)
    """
    # Iterate over the specified columns and convert each one to numeric type
    for column in training_columns:
        h2o_frame[column] = h2o_frame[column].asnumeric()

    return h2o_frame


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

    # Check for duplicate variables in the 'variables' list
    if len(variables) != len(set(variables)):
        raise ValueError("`variables` contains duplicate elements.")

    # Ensure all specified variables are present in the DataFrame
    if not all(var in df.columns for var in variables):
        raise ValueError("`variables` given are not within the input DataFrame.")

    # Select training data from 'set' column if present, otherwise use the entire DataFrame
    if 'set' in df.columns:
        df_train = df[df['set'] == 'training'][[value] + variables]
    else:
        df_train = df[[value] + variables]

    # Default model configuration parameters for H2O AutoML
    default_model_config = {
        'max_models': 10,              # Maximum number of models to train
        'nfolds': 5,                   # Number of cross-validation folds
        'max_mem_size': '16G',         # Maximum memory allocation for H2O
        'include_algos': ['GBM'],      # List of algorithms to include: "GBM", "GLM", "DeepLearning", "DRF", "StackedEnsemble".
        'sort_metric' : "rmse",        # For regression choose between "deviance", "RMSE", "MSE", "MAE", "RMLSE".
        'save_model': True,            # Whether to save the trained model
        'model_name': 'automl',        # Name of the saved model
        'model_path': './',            # Path to save the model
        'seed': seed,                  # Random seed for reproducibility
        'verbose': verbose             # Verbose output
    }

    # Update default model config with any provided user configuration
    if model_config is not None:
        default_model_config.update(model_config)

    # Filter out specific keys that should not be passed directly to H2O AutoML
    default_model_conf = {key: value for key, value in default_model_config.items()
                          if key not in ('max_mem_size', 'save_model', 'model_name', 'model_path', 'verbose')}

    # Determine the number of CPU cores to use, defaulting to all available cores minus one
    n_cores = n_cores or os.cpu_count() - 1

    def train_model():
        """
        Initialize the H2O cluster and train the AutoML model.

        Returns:
            H2OAutoML: Trained AutoML object.
        """
        # Initialize the H2O cluster with specified memory and CPU settings
        init_h2o(n_cores=n_cores, max_mem_size=default_model_config['max_mem_size'])

        # Convert the pandas DataFrame to an H2OFrame for training
        df_h2o = h2o.H2OFrame(df_train)
        response = value  # Specify the target variable
        predictors = [col for col in df_h2o.columns if col != response]  # List of predictor variables
        df_h2o = _convert_h2oframe_to_numeric(df_h2o, predictors)  # Convert H2OFrame columns to numeric if needed

        # Print training progress if verbose is enabled
        if verbose:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Training AutoML...")

        # Initialize and train the AutoML model with the specified configuration
        auto_ml = H2OAutoML(**default_model_conf)
        auto_ml.train(x=predictors, y=response, training_frame=df_h2o)

        # Print the best model information if verbose is enabled
        if verbose:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Best model obtained! - {auto_ml.leaderboard[0, 'model_id']}")

        return auto_ml

    # Retry logic in case of model training failure
    retry_count = 0
    auto_ml = None

    # Retry the model training up to 'max_retries' times if exceptions occur
    while auto_ml is None and retry_count < max_retries:
        retry_count += 1
        try:
            auto_ml = train_model()  # Attempt to train the model
        except Exception as e:
            # Print error message and retry if verbose is enabled
            if verbose:
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Error occurred - {e}")
                print(f"Retrying... (attempt {retry_count} of {max_retries})")
            # Shut down H2O cluster and retry after a short delay
            h2o.cluster().shutdown()
            time.sleep(5)

    # Raise an error if all retries fail
    if auto_ml is None:
        raise RuntimeError(f"Failed to train the model after {max_retries} attempts.")

    # Save the best model if the save_model flag is set
    model = auto_ml.leader
    if default_model_config['save_model']:
        save_h2o(model, default_model_config['model_path'], default_model_config['model_name'])

    return model  # Return the best model


def flaml_train_model(df, value='value', variables=None, model_config=None, seed=7654321, verbose=True):
    """
    Train a machine learning model using FLAML.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing the dataset.
        value (str, optional): Name of the target variable. Default is 'value'.
        variables (list of str, optional): List of feature variables for training.

    Keyword Parameters:
        model_config (dict, optional): Configuration dictionary for model training parameters.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        verbose (bool, optional): Whether to print progress messages. Default is True.

    Returns:
        object: Trained FLAML model.

    Raises:
        ValueError: If `variables` contains duplicates or if any `variables` are not present in the DataFrame.
    """

    # Check for duplicate variable names
    if len(set(variables)) != len(variables):
        raise ValueError("`variables` contains duplicate elements.")

    # Ensure all specified variables are present in the DataFrame
    if not all(var in df.columns for var in variables):
        raise ValueError("`variables` given are not within the input DataFrame.")

    # Extract training data from the 'set' column if present, otherwise use the entire DataFrame
    if 'set' in df.columns:
        df_train = df[df['set'] == 'training'][[value] + variables]
    else:
        df_train = df[[value] + variables]

    # Default model configuration for FLAML
    default_model_config = {
        'time_budget': 90,                     # Total running time in seconds
        'metric': 'rmse',                        # Performance metric (e.g., 'r2', 'mae', 'mse','rmse')
        'estimator_list': ["lgbm"],            # List of estimators: 'lgbm', 'xgboost', 'xgb_limitdepth', 'rf', 'extra_tree'etc.
        'task': 'regression',                  # Task type (e.g., regression or classification)
        'eval_method': 'auto',                 # Evaluation method (e.g., 'auto', 'cv', 'holdout')
        'save_model': True,                    # Whether to save the trained model
        'model_name': 'automl',                # Name of the saved model
        'model_path': './',                    # Directory path to save the model
        'verbose': verbose                     # Whether to print verbose messages
    }

    # Update default model configuration with user-specified settings if provided
    if model_config is not None:
        default_model_config.update(model_config)

    # Filter out keys that should not be passed directly to FLAML
    default_model_conf = {key: value for key, value in default_model_config.items()
                          if key not in ('save_model', 'model_name', 'model_path')}

    # Initialize the FLAML AutoML model
    model = AutoML()

    # Print training progress if verbose is enabled
    if verbose:
        print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), ": Training AutoML...")

    # Train the FLAML model using the specified configuration
    model.fit(X_train=df_train[variables], y_train=df_train[value],
              **default_model_conf, seed=seed)

    # Print the best model details if verbose is enabled
    if verbose:
        print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), ": Best model is",
              model.best_estimator, "with best model parameters of", model.best_config)

    # Save the trained model if the save_model flag is set
    if default_model_config['save_model']:
        save_flaml(model, default_model_config['model_path'], default_model_config['model_name'])

    return model  # Return the trained FLAML model


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
    n_cores = n_cores or os.cpu_count() - 1

    if automl_pkg == "flaml":
        # Train the model using FLAML AutoML
        model = flaml_train_model(df, value, variables, model_config, seed, verbose)
        model._model_type = "flaml"  # Set custom attribute to track model type

    elif automl_pkg == "h2o":
        # Train the model using H2O AutoML
        model = h2o_train_model(df, value, variables, model_config, seed, n_cores, verbose)
        model._model_type = "h2o"  # Set custom attribute to track model type

    return model


def prepare_train_model(df, value, automl_pkg, feature_names, split_method, fraction, model_config, seed=7654321, n_cores=None, verbose=True):
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

    # Create a list of feature names excluding specific columns that are not needed for training
    vars = list(set(feature_names) - set(['date_unix', 'day_julian', 'weekday', 'hour']))

    # Prepare the data for training by processing it according to the specified parameters
    df = prepare_data(df, value, feature_names=vars, split_method=split_method, fraction=fraction, seed=seed)

    # Train the machine learning model using the AutoML package specified
    model = train_model(df, value='value', automl_pkg=automl_pkg, variables=feature_names, model_config=model_config, seed=seed, n_cores=n_cores, verbose=verbose)

    # Return the prepared DataFrame and the trained model
    return df, model


def extract_feature_names(model, importance_ascending=False):
    """
    Extract and sort feature names from the best estimator of an AutoML model, either 'h2o' or 'flaml',
    based on feature importance.

    Parameters:
        model (AutoML): The trained AutoML model object (either from H2O or FLAML framework).
        importance_ascending (bool, optional): Whether to sort feature names in ascending order of importance.
                                               Default is False (descending order).

    Returns:
        list: List of feature names sorted by importance.

    Raises:
        AttributeError: If the model doesn't have identifiable feature names or feature importances.
        TypeError: If the model type is unsupported (not 'h2o' or 'flaml').

    Example Usage:
        # Extract and sort feature names from a trained AutoML model in ascending order of importance
        sorted_feature_names = extract_feature_names(trained_model, importance_ascending=True)
    """

    # If the model is from H2O, extract feature names using the varimp function and sort by importance
    if getattr(model, '_model_type', None) == 'h2o':
        varimp_df = model.varimp(use_pandas=True)
        feature_names = varimp_df['variable'].tolist()
        # Sort by importance based on the importance_ascending flag
        feature_names = varimp_df.sort_values('relative_importance', ascending=importance_ascending)['variable'].tolist()

    # If the model is from FLAML, check for feature names and their importances
    elif getattr(model, '_model_type', None) == 'flaml':
        if hasattr(model, 'feature_name_') and hasattr(model, 'feature_importances_'):
            feature_names = model.feature_name_
            feature_importances = model.feature_importances_
            # Sort feature names based on their importances, controlled by importance_ascending flag
            feature_names = [x for _, x in sorted(zip(feature_importances, feature_names), reverse=not importance_ascending)]
        elif hasattr(model, 'feature_names_in_') and hasattr(model, 'feature_importances_'):
            feature_names = model.feature_names_in_
            feature_importances = model.feature_importances_
            # Sort feature names based on their importances, controlled by importance_ascending flag
            feature_names = [x for _, x in sorted(zip(feature_importances, feature_names), reverse=not importance_ascending)]
        else:
            raise AttributeError("The best estimator does not have identifiable feature names or feature importances.")

    # Raise an error if the model type is neither 'h2o' nor 'flaml'
    else:
        raise TypeError("Unsupported model type. The model must be an H2O or FLAML model.")

    # Return the sorted feature names
    return feature_names


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
            newdata_h2o = _convert_h2oframe_to_numeric(newdata_h2o, extract_feature_names(model))
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


def generate_resampled(df, variables_resample, replace, seed, verbose, weather_df):
    """
    Resamples specified meteorological variables from a weather DataFrame.

    This function is designed to resample meteorological variables (or other specified features) in parallel,
    allowing for the generation of new input data for machine learning models by randomly selecting from the
    available weather data. The resampled values are used to assess the impact of weather variation on the model predictions.

    Parameters:
        df (pandas.DataFrame): The main DataFrame that contains the original dataset.
        variables_resample (list of str): List of meteorological variables (or features) to be resampled.
        replace (bool): Whether to allow replacement during resampling. If True, the same sample can be selected more than once.
        seed (int): Random seed for reproducibility of resampling.
        verbose (bool): Whether to print progress logs during resampling. Progress is printed every 5 iterations.
        weather_df (pandas.DataFrame, optional): Optional DataFrame that contains weather data. If None, uses `df` itself.

    Returns:
        pandas.DataFrame: The input DataFrame (`df`) with resampled meteorological parameters in place.
    """

    # Set random seed for reproducibility
    np.random.seed(seed)

    df[variables_resample] = weather_df[variables_resample].sample(n=len(df), replace=replace).reset_index(drop=True)
    df['seed'] = seed

    return df


def normalise(df, model, feature_names, variables_resample=None, n_samples=300, replace=True,
              aggregate=True, seed=7654321, n_cores=None, weather_df=None, memory_save=False, verbose=True):
    """
    Normalises a dataset using a pre-trained machine learning model with parallel resampling.

    Parameters:
        df (pandas.DataFrame): Input time series data.
        model (object): Pre-trained model (FLAML or H2O).
        feature_names (list of str): List of features to be used for predictions.
        variables_resample (list of str, optional): Features for resampling. Defaults to all features except 'date_unix'.
        n_samples (int, optional): Number of resampling iterations. Default is 300.
        replace (bool, optional): Sample with replacement. Default is True.
        aggregate (bool, optional): Aggregate results across resamples. Default is True.
        seed (int, optional): Random seed. Default is 7654321.
        n_cores (int, optional): CPU cores for parallel processing. Default is all cores minus one.
        weather_df (pandas.DataFrame, optional): External weather data for resampling. Default is None (uses `df`).
        memory_save (bool, optional): Use memory-efficient approach. Default is FALSE.
        verbose (bool, optional): Print progress messages. Default is True.

    Returns:
        pandas.DataFrame: Normalised predictions with optional aggregation.
    """
    # Preprocess input data
    df = process_date(df).pipe(check_data, feature_names, 'value')
    weather_df = weather_df or df
    variables_resample = variables_resample or [var for var in feature_names if var != 'date_unix']

    # Validate input data
    if not all(var in weather_df.columns for var in variables_resample):
        raise ValueError("The weather_df does not contain all variables in `variables_resample`.")

    # Set up seeds and cores
    np.random.seed(seed)
    random_seeds = np.random.choice(np.arange(1000001), size=n_samples, replace=False)
    n_cores = n_cores or (os.cpu_count() - 1)

    if verbose:
        print(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}: Normalising the dataset in parallel.")

    def process_sample(i):
        try:
            df_resampled = generate_resampled(df, variables_resample, replace, random_seeds[i], verbose, weather_df)
            predictions = nm_predict(model, df_resampled)
            return pd.DataFrame({
                'date': df_resampled['date'],
                'observed': df_resampled['value'],
                'normalised': predictions,
                'seed': random_seeds[i]
            })
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    # Choose the memory-saving method or process all resamples at once
    if memory_save:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_cores) as executor:
            generated_dfs = [f.result() for f in concurrent.futures.as_completed(
                [executor.submit(process_sample, i) for i in range(n_samples)])]
        df_result = pd.concat([df for df in generated_dfs if df is not None])
    else:
        df_resampled_list = Parallel(n_jobs=n_cores)(delayed(generate_resampled)(
            df, variables_resample, replace, random_seeds[i], False, weather_df) for i in range(n_samples))
        df_all_resampled = pd.concat(df_resampled_list, ignore_index=True)
        predictions = nm_predict(model, df_all_resampled)
        df_result = pd.DataFrame({
            'date': df_all_resampled['date'],
            'observed': df_all_resampled['value'],
            'normalised': predictions,
            'seed': df_all_resampled['seed']
        })

    # Aggregate or pivot results
    if aggregate:
        if verbose:
            print(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}: Aggregating {n_samples} predictions.")
        df_result = df_result.groupby('date').mean()[['observed', 'normalised']]
    else:
        df_result = pd.concat([
            df_result.drop_duplicates(subset=['date']).set_index('date')[['observed']],
            df_result.pivot(index='date', columns='seed', values='normalised')
        ], axis=1)

    if verbose:
        print(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}: Finished processing {n_samples} predictions.")

    return df_result


def do_all(df=None, value=None, automl_pkg='flaml', feature_names=None, variables_resample=None, split_method='random', fraction=0.75,
           model_config=None, n_samples=300, seed=7654321, n_cores=None, aggregate=True, weather_df=None, memory_save=False, verbose=True):
    """
    Conducts data preparation, model training, and normalisation, returning the transformed dataset and model statistics.

    This function performs a complete pipeline that includes preparing the dataset, training a machine learning model,
    and applying normalisation through resampling of meteorological (or other) variables. It uses an AutoML package
    (such as FLAML or H2O) for model training and returns the normalised dataset as well as model performance statistics.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing the dataset. It should include the features and target variable.
        value (str): Name of the target variable in the dataset, which the model will predict.
        automl_pkg (str, optional): The AutoML package to use for training the model. Options include 'flaml' and 'h2o'. Default is 'flaml'.
        feature_names (list of str): List of feature names to be used as input for the model.
        variables_resample (list of str, optional): List of variables to be resampled during normalisation. Default is None.
        split_method (str, optional): The method used to split the data for training. Options are:
            - 'random': Randomly splits the data into training and test sets.
            - 'time_series': Splits the data in chronological order for time series analysis.
            Default is 'random'.
        fraction (float, optional): The fraction of the dataset to use for training. The rest will be used for testing. Default is 0.75 (75% for training).
        model_config (dict, optional): A dictionary containing configuration parameters for model training, such as hyperparameters. Default is None.
        n_samples (int, optional): The number of resampled datasets to generate for normalisation. Default is 300.
        seed (int, optional): Random seed for ensuring reproducibility across different runs. Default is 7654321.
        n_cores (int, optional): Number of CPU cores to use for parallel processing. If None, it uses all available cores minus one. Default is None.
        aggregate (bool, optional): Whether to aggregate the results across all samples. Default is True.
        weather_df (pandas.DataFrame, optional): A DataFrame containing weather data or external features for resampling. If None, the input `df` is used. Default is None.
        memory_save (bool, optional): Use memory-efficient approach. Default is FALSE.
        verbose (bool, optional): Whether to print progress messages during the process. Default is True.

    Returns:
        tuple:
            - result (pandas.DataFrame): The normalised dataset with predicted and resampled values.
            - mod_stats (pandas.DataFrame): DataFrame containing model performance statistics, such as accuracy or error metrics.

    Example:
        >>> df = pd.read_csv('timeseries_data.csv')
        >>> value = 'target'
        >>> feature_names = ['feature1', 'feature2', 'feature3']
        >>> variables_resample = ['feature1', 'feature2']
        >>> result, mod_stats = do_all(df, value=value, automl_pkg='flaml', feature_names=feature_names, variables_resample=variables_resample)
    """

    # Step 1: Prepare the dataset and train the model using the specified AutoML package (e.g., FLAML or H2O).
    df, model = prepare_train_model(
        df=df, value=value, automl_pkg=automl_pkg, feature_names=feature_names,
        split_method=split_method, fraction=fraction, model_config=model_config,
        seed=seed, verbose=verbose
    )

    # Step 2: Determine the number of CPU cores to use. If not provided, use all cores minus one.
    n_cores = n_cores or os.cpu_count() - 1

    # Step 3: Perform normalisation of the data using the trained model and external weather data (if provided).
    result = normalise(
        df=df, model=model, feature_names=feature_names, variables_resample=variables_resample,
        n_samples=n_samples, aggregate=aggregate, n_cores=n_cores, seed=seed,
        weather_df=weather_df, memory_save=memory_save, verbose=verbose
    )

    # Step 4: Compute and return model statistics, such as accuracy or error metrics, based on the test set.
    mod_stats = modStats(df, model)

    # Step 5: Return the final normalised dataset and the model statistics.
    return result, mod_stats


def do_all_unc(df=None, value=None, automl_pkg='flaml', feature_names=None, variables_resample=None, split_method='random',
               fraction=0.75, model_config=None, n_samples=300, n_models=10, confidence_level=0.95, seed=7654321,
               n_cores=None, weather_df=None, memory_save=False, verbose=True):
    """
    Performs uncertainty quantification by training multiple models with different random seeds and calculates statistical metrics.

    This function is used to estimate the uncertainty in model predictions by training multiple models using different random
    seeds, then calculating various statistical measures such as the mean, standard deviation, median, and confidence bounds
    across these models' predictions. The process involves normalising the input data based on weather conditions (or other
    features), training the models, and aggregating the predictions to provide an overall estimate of uncertainty.

    Parameters:
        df (pandas.DataFrame): Input dataframe containing the time series data. Must include the target variable and features.
        value (str): Column name of the target variable to be predicted by the model.
        feature_names (list of str): List of feature column names that will be used as input to the model.
        variables_resample (list of str): List of sampled feature names for normalisation.
        split_method (str, optional): Method to split the data. Options include:
            - 'random': Randomly splits the data.
            Default is 'random'.
        fraction (float, optional): Fraction of the dataset to use for training. The remaining fraction will be used for testing. Default is 0.75 (75% for training).
        model_config (dict, optional): A configuration dictionary containing hyperparameters and other settings for model training.
        n_samples (int, optional): Number of samples for normalisation. Default is 300.
        n_models (int, optional): Number of models to train for uncertainty quantification. Each model is trained with a different random seed. Default is 10.
        confidence_level (float, optional): Confidence level for the uncertainty bounds (e.g., 0.95 corresponds to 95% confidence intervals). Default is 0.95.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        n_cores (int, optional): Number of CPU cores to use for parallel processing. If not specified, defaults to total CPU cores minus one.
        weather_df (pandas.DataFrame, optional): DataFrame containing weather data for resampling. If None, the input `df` is used. Default is None.
        memory_save (bool, optional): Use memory-efficient approach. Default is FALSE.
        verbose (bool, optional): Whether to print progress messages during the process. Default is True.

    Returns:
        tuple:
            - df_dew (pandas.DataFrame): A dataframe containing the following columns:
                - 'observed': The actual observed values from the input data.
                - 'mean': The mean prediction across the multiple models.
                - 'std': The standard deviation of the predictions across models, representing the uncertainty.
                - 'median': The median prediction across models.
                - 'lower_bound': The lower bound of the confidence interval.
                - 'upper_bound': The upper bound of the confidence interval.
                - 'weighted': The weighted average of the model predictions, using R-squared as weights.
            - mod_stats (pandas.DataFrame): Dataframe containing model statistics such as R-squared, calculated for each model.

    Example:
        >>> df = pd.read_csv('timeseries_data.csv')
        >>> value = 'target'
        >>> feature_names = ['feature1', 'feature2', 'feature3']
        >>> variables_resample = ['feature1', 'feature2']
        >>> df_dew, mod_stats = do_all_unc(df, value=value, automl_pkg='flaml', feature_names=feature_names, variables_resample=variables_resample)
    """

    # Step 1: Generate random seeds for training multiple models
    np.random.seed(seed)
    random_seeds = np.random.choice(np.arange(1000001), size=n_models, replace=False).tolist()

    df_dew_list = []
    mod_stats_list = []

    # Step 2: Determine the number of CPU cores to use
    nn_cores = n_cores or os.cpu_count() - 1

    # Record start time for ETA calculation
    start_time = time.time()
    if model_config is None:
        model_config = {'save_model': False}

    # Step 3: Train multiple models with different random seeds
    for i, seed in enumerate(random_seeds):
        # Train and normalise data using do_all function
        df_dew0, mod_stats0 = do_all(df, value=value, automl_pkg=automl_pkg, feature_names=feature_names,
                                     variables_resample=variables_resample,
                                     split_method=split_method, fraction=fraction,
                                     model_config=model_config,
                                     n_samples=n_samples, seed=seed, n_cores=n_cores,
                                     weather_df=weather_df, memory_save=memory_save, verbose=False)

        # Rename columns to reflect the current random seed
        df_dew0.rename(columns={'normalised': f'normalised_{seed}'}, inplace=True)
        df_dew0 = df_dew0[['observed', f'normalised_{seed}']]
        df_dew_list.append(df_dew0)

        # Add the random seed to model statistics
        mod_stats0['seed'] = seed
        mod_stats_list.append(mod_stats0)

        # Print progress and estimated time remaining
        if verbose:
            elapsed_time = time.time() - start_time
            progress_percent = (i + 1) / n_models * 100
            remaining_time = elapsed_time / (i + 1) * (n_models - (i + 1))
            if remaining_time < 60:
                remaining_str = "ETA: {:.2f} seconds".format(remaining_time)
            elif remaining_time < 3600:
                remaining_str = "ETA: {:.2f} minutes".format(remaining_time / 60)
            else:
                remaining_str = "ETA: {:.2f} hours".format(remaining_time / 3600)
            print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                  ": Progress: {:.2f}% (Model {}/{})... {}".format(progress_percent, i + 1, n_models, remaining_str))

    # Step 4: Combine all normalized predictions into a single DataFrame
    df_dew = pd.concat(df_dew_list, axis=1)
    df_dew = df_dew.loc[:, ~df_dew.columns.duplicated()]

    # Combine all model statistics
    mod_stats = pd.concat(mod_stats_list, ignore_index=True)

    # Step 5: Calculate statistical metrics across all model predictions
    predictions = df_dew.iloc[:, 1:n_models + 1]
    df_dew['mean'] = predictions.mean(axis=1)
    df_dew['std'] = predictions.std(axis=1)
    df_dew['median'] = predictions.median(axis=1)
    df_dew['lower_bound'] = predictions.quantile((1 - confidence_level) / 2, axis=1)
    df_dew['upper_bound'] = predictions.quantile(1 - (1 - confidence_level) / 2, axis=1)

    # Step 6: Calculate weighted R2 scores for each model
    test_stats = mod_stats[mod_stats['set'] == 'testing']
    test_stats.loc[:, 'R2'] = test_stats['R2'].replace([np.inf, -np.inf], np.nan)

    #test_stats['R2'] = test_stats['R2'].replace([np.inf, -np.inf], np.nan)
    normalised_R2 = (test_stats['R2'] - test_stats['R2'].min()) / (test_stats['R2'].max() - test_stats['R2'].min())
    weighted_R2 = normalised_R2 / normalised_R2.sum()

    # Apply weighted R2 to predictions to calculate a weighted average
    df_dew['weighted'] = (predictions.values * weighted_R2.values[np.newaxis, :]).sum(axis=1)

    return df_dew, mod_stats


def decom_emi(df=None, model=None, value='value', automl_pkg='flaml', feature_names=None, split_method='random', fraction=0.75,
              model_config=None, n_samples=300, seed=7654321, n_cores=None, memory_save= False, verbose=True):
    """
    Decomposes a time series into different components using machine learning models.

    Parameters:
        df (pandas.DataFrame): Input dataframe containing the time series data.
        model (object, optional): Pre-trained model to use for decomposition. If None, a new model will be trained. Default is None.
        value (str): Column name of the target variable. Default is 'value'.
        feature_names (list of str): List of feature column names used for modeling and decomposition.
        split_method (str, optional): Method to split the data ('random' or other methods). Default is 'random'.
        fraction (float, optional): Fraction of data to be used for training. Default is 0.75.
        model_config (dict, optional): Configuration dictionary for model training parameters.
        n_samples (int, optional): Number of samples for normalization. Default is 300.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        n_cores (int, optional): Number of cores to be used. Default is total CPU cores minus one.
        memory_save (bool, optional): Use memory-efficient approach. Default is FALSE.
        verbose (bool, optional): Whether to print progress messages. Default is True.

    Returns:
        result (pandas.DataFrame): Dataframe with decomposed components.
    """

    # If no model is provided, train a new model
    if model is None:
        df, model = prepare_train_model(df, value, automl_pkg, feature_names, split_method, fraction, model_config, seed, verbose)

    # If the model is provided, ensure that the 'value' column exists in the dataframe; otherwise, prepare the data
    elif value not in df.columns:
        # Prepare the data if the value column is not in the dataframe
        vars = list(set(feature_names) - set(['date_unix', 'day_julian', 'weekday', 'hour']))
        df = prepare_data(df, value, feature_names=vars, split_method=split_method, fraction=fraction, seed=seed)

    # Initialize the dataframe for storing decomposed components
    df_dew = df[['date', 'value']].set_index('date').rename(columns={value: 'observed'})

    # Set default number of CPU cores if not specified
    n_cores = n_cores or os.cpu_count() - 1

    # Initialize a list to store variables to exclude
    vars_to_exclude = ['date_unix', 'day_julian', 'weekday', 'hour']
    vars_intersect = sorted(set(vars_to_exclude).intersection(feature_names), key=vars_to_exclude.index)

    # Start decomposing the time series by iterating over the excluded features
    start_time = time.time()

    for i, var_to_exclude in enumerate(['base']+vars_intersect):
        if verbose:
            if i == 0:
                print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), f": Subtracting {var_to_exclude}...")
            else:
                # Estimate remaining time for the decomposition process
                elapsed_time = time.time() - start_time
                remaining_time = elapsed_time / i * (len(vars_intersect) - i)
                if remaining_time < 60:
                    eta_str = "ETA: {:.2f} seconds".format(remaining_time)
                elif remaining_time < 3600:
                    eta_str = "ETA: {:.2f} minutes".format(remaining_time / 60)
                else:
                    eta_str = "ETA: {:.2f} hours".format(remaining_time / 3600)
                print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), f": Subtracting {var_to_exclude}... {eta_str}")

        # Exclude the current variable from feature_names
        var_names = list(set(feature_names) - {var_to_exclude})

        # Normalize the data excluding the current variable
        success = False
        retries = 3
        while not success and retries > 0:
            try:
                df_dew_temp = normalise(df, model, feature_names=feature_names, variables_resample=var_names,
                                        n_samples=n_samples, n_cores=n_cores, seed=seed, memory_save=memory_save, verbose=False)
                df_dew[var_to_exclude] = df_dew_temp['normalised']
                success = True
            except Exception as e:
                print(f"Error during normalization for variable '{var_to_exclude}': {e}")
                retries -= 1
                if retries > 0:
                    print(f"Retrying... {retries} attempts left.")
                    time.sleep(10)
                else:
                    print("Failed after 3 attempts. Moving to the next variable.")
                    df_dew[var_to_exclude] = np.nan

    result = df_dew.copy()
    for i, var_current in enumerate(vars_intersect):
        if i == 0:
            result[var_current] = df_dew[var_current] - df_dew['base'] + df_dew['base'].mean()
        else:
            var_previous = vars_intersect[i - 1]
            result[var_current] = df_dew[var_current] - df_dew[var_previous]

    if vars_intersect:
        result['deweathered'] = df_dew[vars_intersect[-1]]

    result['emi_noise'] = df_dew['base'] - df_dew['base'].mean()

    return result


def decom_met(df=None, model=None, value='value', automl_pkg='flaml', feature_names=None, split_method='random', fraction=0.75,
              model_config=None, n_samples=300, seed=7654321, importance_ascending=False, n_cores=None, memory_save=False, verbose=True):
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
        n_samples (int, optional): Number of samples for normalization. Default is 300.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        importance_ascending (bool, optional): Sort order for feature importances. Default is False.
        n_cores (int, optional): Number of cores to be used. Default is total CPU cores minus one.
        memory_save (bool, optional): Use memory-efficient approach. Default is FALSE.
        verbose (bool, optional): Whether to print progress messages. Default is True.

    Returns:
        result (pandas.DataFrame): Dataframe with decomposed components.
    """

    # Train the model if it is not provided, using the specified parameters for training (AutoML)
    if model is None:
        df, model = prepare_train_model(df, value, automl_pkg, feature_names, split_method, fraction, model_config, seed, verbose=verbose)

    # If the model is provided, ensure that the 'value' column exists in the dataframe; otherwise, prepare the data
    elif 'value' not in df.columns:
        vars = list(set(feature_names) - set(['date_unix', 'day_julian', 'weekday', 'hour']))
        df = prepare_data(df, value, feature_names=vars, split_method=split_method, fraction=fraction, seed=seed)

    # Extract and sort the feature importances from the model in the specified order (ascending or descending)
    feature_names_sorted = extract_feature_names(model, importance_ascending=importance_ascending)

    # Initialize the dataframe for storing decomposed components (observed value and deweathered data)
    df_dew = df[['date', 'value']].set_index('date').rename(columns={'value': 'observed'})

    # Create a list of features to exclude, starting with 'deweathered'
    met_list = ['deweathered'] + [item for item in feature_names_sorted if item not in ['hour', 'weekday', 'day_julian', 'date_unix']]

    # Filter out time-related features from the feature list
    var_names = [item for item in feature_names_sorted if item not in ['hour', 'weekday', 'day_julian', 'date_unix']]

    # Set the number of CPU cores to use, defaulting to all cores minus one
    n_cores = n_cores or os.cpu_count() - 1

    # Track the start time for calculating the ETA during the loop
    start_time = time.time()

    # Loop through each feature in the met_list, excluding one feature at a time
    for i, var_to_exclude in enumerate(met_list):
        if verbose:
            # Calculate the estimated time remaining for the process
            elapsed_time = time.time() - start_time
            eta = (elapsed_time / (i + 1)) * (len(met_list) - (i + 1))
            eta_str = f"ETA: {eta/60:.2f} minutes" if eta > 60 else f"ETA: {eta:.2f} seconds"

            # Print progress with an ETA
            print(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}: Subtracting {var_to_exclude}... {eta_str}")

        # Exclude the current variable from the feature list for resampling
        var_names = [v for v in var_names if v != var_to_exclude]

        # Normalize the data by excluding the current variable from the resampling process
        df_dew_temp = normalise(df, model, feature_names=feature_names, variables_resample=var_names,
                                n_samples=n_samples, n_cores=n_cores, seed=seed, memory_save=memory_save, verbose=False)

        # Add the normalized data to the dataframe as a new decomposed component
        df_dew[var_to_exclude] = df_dew_temp['normalised']

    # Create a copy of the dataframe for adjusting decomposed components
    result = df_dew.copy()

    # Adjust each decomposed component, subtracting previous components to ensure they are weather-independent
    for i, param in enumerate([item for item in feature_names_sorted if item not in ['hour', 'weekday', 'day_julian', 'date_unix']]):
        if i > 0:
            # Subtract the previous component from the current component
            result[param] = df_dew[param] - df_dew[met_list[i - 1]]
        else:
            # For the first component ('deweathered'), subtract it from the observed value
            result[param] = df_dew[param] - df_dew['deweathered']

    # Calculate the residual noise component by subtracting the last decomposed component from the observed value
    result['met_noise'] = df_dew['observed'] - df_dew[met_list[-1]]

    # Return the dataframe containing the decomposed components
    return result


def rolling(df=None, model=None, value='value', automl_pkg='flaml', feature_names=None, variables_resample=None, split_method='random', fraction=0.75,
            model_config=None, n_samples=300, window_days=14, rolling_every=7, seed=7654321, n_cores=None, memory_save=False, verbose=True):
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
        memory_save (bool, optional): Use memory-efficient approach. Default is FALSE.
        verbose (bool, optional): Whether to print progress messages. Default is True.

    Returns:
        combined_results (pandas.DataFrame): Dataframe containing observed and rolling normalized results.
        mod_stats (pandas.DataFrame): Dataframe with model statistics.
    """

    # If no model is provided, train a new model
    if model is None:
        df, model = prepare_train_model(df, value, automl_pkg, feature_names, split_method, fraction, model_config, seed, verbose)

    # If the model is provided, ensure that the 'value' column exists in the dataframe; otherwise, prepare the data
    elif value not in df.columns:
        # Prepare the data if the value column is not in the dataframe
        vars = list(set(feature_names) - set(['date_unix', 'day_julian', 'weekday', 'hour']))
        df = prepare_data(df, value, feature_names=vars, split_method=split_method, fraction=fraction, seed=seed)

    # Default logic to determine the number of CPU cores for parallel computation
    n_cores = n_cores or os.cpu_count() - 1

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
        retry = False  # Set retry flag
        for attempt in range(2):  # Attempt loop (maximum 2 tries)
            try:
                # Filter the data for the current rolling window (from start date `ds` to window_days length)
                dfa = df[df['date_d'] >= ds.date()]
                dfa = dfa[dfa['date_d'] < (dfa['date_d'].min() + pd.DateOffset(days=window_days)).date()]

                # Normalize the data for the current rolling window
                dfar = normalise(dfa, model, feature_names=feature_names, variables_resample=variables_resample,
                                 n_samples=n_samples, n_cores=n_cores, seed=seed, memory_save=memory_save, verbose=False)

                # Rename the 'normalised' column to indicate the rolling window index
                dfar.rename(columns={'normalised': 'rolling_' + str(i)}, inplace=True)

                # Merge the normalized results of the current window with the overall result DataFrame
                combined_results = pd.concat([combined_results, dfar['rolling_' + str(i)]], axis=1)

                # If verbose is enabled, print progress every 10 rolling windows
                if verbose and (i % 10 == 0):
                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Rolling window {i} from {dfa['date'].min().strftime('%Y-%m-%d')} to {(dfa['date'].max()).strftime('%Y-%m-%d')}")

                break  # Break out of the attempt loop if successful

            except Exception as e:
                # Handle any errors during the normalization process and retry once
                if verbose:
                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Error during normalization for rolling window {i} from {dfa['date'].min().strftime('%Y-%m-%d')} to {(dfa['date'].max()).strftime('%Y-%m-%d')}: {str(e)}")

                if attempt == 1:  # If second attempt also fails, stop retrying
                    if verbose:
                        print(f"Rolling window {i} failed after retrying.")
                else:
                    if verbose:
                        print("Retrying once...")

    # Return the final DataFrame containing observed and rolling normalized values, along with model statistics
    return combined_results


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
    # Default list of statistics to calculate if none is provided
    if statistic is None:
        statistic = ["n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA", "R2"]

    # Nested function to calculate statistics for a specific set
    def calculate_stats(df, set_name=None):
        if set_name is not None:
            # Filter the DataFrame based on the 'set' column (e.g., 'training', 'testing', etc.)
            if 'set' in df.columns:
                df = df[df['set'] == set_name]
            else:
                raise ValueError(f"The DataFrame does not contain the 'set' column but 'set' parameter was provided as '{set_name}'.")

        # Generate model predictions and assign them to a new column 'value_predict'
        df = df.assign(value_predict=nm_predict(model,df))

        # Calculate the statistics using the Stats function (presumably custom) and return the result
        df_stats = Stats(df, mod="value_predict", obs="value", statistic=statistic).assign(set=set_name)
        return df_stats

    # If 'set' parameter is not provided, calculate statistics for each available set in the DataFrame
    if set is None:
        if 'set' in df.columns:
            # Get all unique values in the 'set' column (e.g., 'training', 'testing')
            sets = df['set'].unique()

            # Calculate statistics for each set
            stats_list = [calculate_stats(df, s) for s in sets]

            # Add statistics for the whole dataset (combining all sets)
            df_all = df.copy()
            df_all['set'] = 'all'  # Mark the entire dataset with 'all'
            stats_list.append(calculate_stats(df_all, 'all'))

            # Concatenate all statistics into a single DataFrame
            df_stats = pd.concat(stats_list, ignore_index=True)
        else:
            raise ValueError("The DataFrame does not contain the 'set' column and 'set' parameter was not provided.")
    else:
        # Calculate statistics for the specified set (e.g., 'training' or 'testing')
        df_stats = calculate_stats(df, set)

    return df_stats


def Stats(df, mod, obs, statistic=None):
    """
    Calculates specified statistics based on provided data.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.
        statistic (list): List of statistics to calculate. If None, default statistics will be calculated.

    Returns:
        DataFrame: DataFrame containing calculated statistics.
    """

    # If no specific statistics are provided, default to calculating a predefined list of statistics
    if statistic is None:
        statistic = ["n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA", "R2"]

    # Dictionary to store calculated statistics
    res = {}

    # Calculate each specified statistic if it is in the list of statistics to be calculated
    if "n" in statistic:
        res["n"] = n(df, mod, obs)  # Calculate the number of observations
    if "FAC2" in statistic:
        res["FAC2"] = FAC2(df, mod, obs)  # Calculate the Factor of Two (FAC2) statistic
    if "MB" in statistic:
        res["MB"] = MB(df, mod, obs)  # Calculate the Mean Bias (MB)
    if "MGE" in statistic:
        res["MGE"] = MGE(df, mod, obs)  # Calculate the Mean Gross Error (MGE)
    if "NMB" in statistic:
        res["NMB"] = NMB(df, mod, obs)  # Calculate the Normalized Mean Bias (NMB)
    if "NMGE" in statistic:
        res["NMGE"] = NMGE(df, mod, obs)  # Calculate the Normalized Mean Gross Error (NMGE)
    if "RMSE" in statistic:
        res["RMSE"] = RMSE(df, mod, obs)  # Calculate the Root Mean Square Error (RMSE)
    if "r" in statistic:
        # Calculate the correlation coefficient (r) and its p-value
        res["r"] = r(df, mod, obs)[0]  # Store the correlation coefficient
        p_value = r(df, mod, obs)[1]  # Extract the p-value from the correlation calculation

        # Assign significance level based on p-value
        if p_value >= 0.1:
            res["p_level"] = ""  # Not significant
        elif p_value < 0.1 and p_value >= 0.05:
            res["p_level"] = "+"  # Marginal significance
        elif p_value < 0.05 and p_value >= 0.01:
            res["p_level"] = "*"  # Significant
        elif p_value < 0.01 and p_value >= 0.001:
            res["p_level"] = "**"  # Highly significant
        else:
            res["p_level"] = "***"  # Extremely significant

    if "COE" in statistic:
        res["COE"] = COE(df, mod, obs)  # Calculate the Coefficient of Efficiency (COE)
    if "IOA" in statistic:
        res["IOA"] = IOA(df, mod, obs)  # Calculate the Index of Agreement (IOA)
    if "R2" in statistic:
        res["R2"] = R2(df, mod, obs)  # Calculate the R-squared value

    # Create a dictionary to store all calculated results
    results = {
        'n': res['n'],
        'FAC2': res['FAC2'],
        'MB': res['MB'],
        'MGE': res['MGE'],
        'NMB': res['NMB'],
        'NMGE': res['NMGE'],
        'RMSE': res['RMSE'],
        'r': res['r'],
        'p_level': res['p_level'],
        'COE': res['COE'],
        'IOA': res['IOA'],
        'R2': res['R2']
    }

    # Convert the results dictionary into a DataFrame for easier output
    results = pd.DataFrame([results])

    return results  # Return the DataFrame containing the calculated statistics


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
    n_cores = n_cores or os.cpu_count() - 1

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
        model: AutoML model object. The model used to compute predictions.
        X_train (DataFrame): Input DataFrame containing the training data. This is the dataset used for calculating partial dependence.
        variable (str): Name of the feature to compute partial dependence plot for. Specifies which feature to focus on.
        training_only (bool, optional): If True, computes partial dependence plot only for the training set. Default is True.
                                       Currently, this parameter is not being utilized.

    Returns:
        DataFrame: DataFrame containing the computed partial dependence plot for the specified feature.
                   The DataFrame will include the feature's values, the mean of predictions (PDP mean),
                   and the standard deviation (PDP std).
    """

    # Compute partial dependence using the trained model on the specified feature.
    # `kind='individual'` generates Individual Conditional Expectation (ICE) curves.
    results = partial_dependence(estimator=model, X=X_train, features=variable, kind='individual')

    # Create a DataFrame for storing the feature values and PDP statistics (mean and standard deviation).
    df_predict = pd.DataFrame({"value": results['grid_values'][0],  # Feature values for which PDP is calculated.
                               "pdp_mean": np.mean(results['individual'][0], axis=0),  # Mean of the individual predictions for the feature.
                               'pdp_std': np.std(results['individual'][0], axis=0)})  # Standard deviation of predictions for the feature.

    # Add the variable name to the DataFrame.
    df_predict["variable"] = variable

    # Reorganize the DataFrame columns for consistency in order: feature name, value, mean, and std.
    df_predict = df_predict[["variable", "value", "pdp_mean", "pdp_std"]]

    # Return the DataFrame containing the partial dependence plot data for the specified feature.
    return df_predict


def scm_all(df, poll_col, code_col, control_pool, cutoff_date, n_cores=None):
    """
    Performs Synthetic Control Method (SCM) in parallel for multiple treatment targets.

    Parameters:
        df (DataFrame): Input DataFrame containing the dataset. It should contain both the treatment and control group data.
        poll_col (str): Name of the column containing the poll data (the dependent variable for SCM).
        code_col (str): Name of the column containing the group codes (the identifier for different regions, countries, etc.).
        control_pool (list): List of codes that make up the control pool. These are the units used to create synthetic controls.
        cutoff_date (str): Date to split the data into pre-treatment and post-treatment periods.
        n_cores (int, optional): Number of CPU cores to use for parallel processing. If not provided, defaults to the total number of CPU cores minus one.

    Returns:
        DataFrame: DataFrame containing the synthetic control results for all treatment targets.

    Example Usage:
        # Perform SCM in parallel for multiple treatment targets
        synthetic_all = scm_all(df, poll_col='Poll', code_col='Code',
                                     control_pool=['A', 'B', 'C'], cutoff_date='2020-01-01', n_cores=4)
    """

    # If the number of cores is not provided, set it to the total number of CPU cores minus one.
    n_cores = n_cores or os.cpu_count() - 1

    # Extract the unique treatment targets from the dataset based on the code_col (e.g., different regions or entities).
    treatment_pool = df[code_col].unique()

    # Perform SCM for each treatment target in parallel using all available CPU cores.
    # The `Parallel` and `delayed` functions from the `joblib` library enable parallel processing for faster computation.
    synthetic_all = pd.concat(Parallel(n_jobs=n_cores)(
        delayed(scm)(
            df=df,
            poll_col=poll_col,
            code_col=code_col,
            treat_target=code,   # Each unique treatment code is processed individually.
            control_pool=control_pool,
            cutoff_date=cutoff_date
        ) for code in treatment_pool  # Loop over all treatment codes.
    ))

    # Return the combined DataFrame with results for all treatment targets.
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
    result = (df[df[code_col] == treat_target][['date', code_col, poll_col]]
            .assign(synthetic=sc)).set_index('date')
    result['effects'] = result[poll_col] - result['synthetic']

    return result


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
    result = (df[df[code_col] == treat_target][['date', code_col, poll_col]]
            .assign(synthetic=nm_predict(model, dfp)))  # Assign the predicted synthetic values
    result = result.set_index('date')  # Set the 'date' column as the index

    # Calculate the treatment effects by subtracting the synthetic values from the actual observed values
    result['effects'] = result[poll_col] - result['synthetic']

    # Return the final DataFrame containing the synthetic control values and estimated treatment effects
    return result


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
    n_cores = n_cores or os.cpu_count() - 1

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
