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

def ts_decom(df, value, feature_names, split_method='random', time_budget=60, metric='r2',
             estimator_list=["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"], task='regression',
             n_samples=300, fraction=0.75, seed=7654321, n_cores=-1):
    """
    Decomposes a time series into different components using machine learning models.

    This function prepares the data, trains a machine learning model using AutoML, and decomposes
    the time series data into various components. The decomposition is based on the contribution
    of different features to the target variable. It returns the decomposed data and model statistics.

    Parameters:
        df (pd.DataFrame): Input dataframe containing the time series data.
        value (str): Column name of the target variable.
        feature_names (list of str): List of feature column names.
        split_method (str, optional): Method to split the data ('random' or other methods). Default is 'random'.
        time_budget (int, optional): Time budget for the AutoML training in seconds. Default is 60.
        metric (str, optional): Metric to evaluate the model ('r2', 'mae', etc.). Default is 'r2'.
        estimator_list (list of str, optional): List of estimators to be used in AutoML. Default is ["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"].
        task (str, optional): Task type ('regression' or 'classification'). Default is 'regression'.
        n_samples (int, optional): Number of samples for normalization. Default is 300.
        fraction (float, optional): Fraction of data to be used for training. Default is 0.75.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        n_cores (int, optional): Number of cores to be used (-1 for all available cores). Default is -1.

    Returns:
        df_dewc (pd.DataFrame): Dataframe with decomposed components.
        mod_stats (pd.DataFrame): Dataframe with model statistics.

    Example:
        >>> df = pd.read_csv('timeseries_data.csv')
        >>> value = 'target'
        >>> feature_names = ['feature1', 'feature2', 'feature3']
        >>> df_dewc, mod_stats = ts_decom(df, value, feature_names)
    """

    # Prepare the data for modeling
    df = prepare_data(df, value=value, feature_names=feature_names, split_method=split_method, fraction=fraction, seed=seed)

    # Train the model using AutoML
    automl = train_model(df, value='value', variables=feature_names, time_budget=time_budget, metric=metric, task=task, seed=seed)

    # Gather model statistics for testing, training, and all data
    mod_stats = pd.concat([modStats(df, automl, set='testing'),
                           modStats(df, automl, set='training'),
                           modStats(df.assign(set="all"), automl, set='all')])

    # Initialize the dataframe for decomposed components
    var_names = feature_names
    df_dew = df[['date', 'value']].set_index('date').rename(columns={'value': 'Observed'})

    # Decompose the time series by excluding different features
    for var_to_exclude in ['all', 'date_unix', 'day_julian', 'weekday', 'hour']:
        var_names = list(set(var_names) - set([var_to_exclude]))
        df_dew_temp = normalise(automl, df, feature_names=feature_names, variables=var_names,
                                n_samples=n_samples, n_cores=n_cores, seed=seed)

        df_dew[var_to_exclude] = df_dew_temp['Normalised']

    # Adjust the decomposed components to create deweathered values
    df_dewc = df_dew.copy()
    df_dewc['hour'] = df_dew['hour'] - df_dew['weekday']
    df_dewc['weekday'] = df_dew['weekday'] - df_dew['day_julian']
    df_dewc['day_julian'] = df_dew['day_julian'] - df_dew['date_unix']
    df_dewc['date_unix'] = df_dew['date_unix'] - df_dew['all'] + df_dew['hour'].mean()
    df_dewc['Deweathered'] = df_dew['hour']

    return df_dewc, mod_stats


def met_rolling(df, value, feature_names, split_method='random', time_budget=60, metric='r2',
                estimator_list=["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"], task='regression',
                n_samples=300, window_days=14, rollingevery=2, fraction=0.75, seed=7654321, n_cores=-1):
    """
    Applies a rolling window approach to decompose the time series into different components using machine learning models.

    This function prepares the data, trains a machine learning model using AutoML, and applies a rolling window approach
    to decompose the time series data into various components. The decomposition is based on the contribution of different
    features to the target variable. It returns the decomposed data and model statistics.

    Parameters:
        df (pd.DataFrame): Input dataframe containing the time series data.
        value (str): Column name of the target variable.
        feature_names (list of str): List of feature column names.
        split_method (str, optional): Method to split the data ('random' or other methods). Default is 'random'.
        time_budget (int, optional): Time budget for the AutoML training in seconds. Default is 60.
        metric (str, optional): Metric to evaluate the model ('r2', 'mae', etc.). Default is 'r2'.
        estimator_list (list of str, optional): List of estimators to be used in AutoML. Default is ["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"].
        task (str, optional): Task type ('regression' or 'classification'). Default is 'regression'.
        n_samples (int, optional): Number of samples for normalization. Default is 300.
        window_days (int, optional): Number of days for the rolling window. Default is 14.
        rollingevery (int, optional): Rolling interval in days. Default is 2.
        fraction (float, optional): Fraction of data to be used for training. Default is 0.75.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        n_cores (int, optional): Number of cores to be used (-1 for all available cores). Default is -1.

    Returns:
        df_dew (pd.DataFrame): Dataframe with decomposed components including mean and standard deviation of the rolling window.
        mod_stats (pd.DataFrame): Dataframe with model statistics.

    Example:
        >>> df = pd.read_csv('timeseries_data.csv')
        >>> value = 'target'
        >>> feature_names = ['feature1', 'feature2', 'feature3']
        >>> df_dew, mod_stats = met_rolling(df, value, feature_names, window_days=14, rollingevery=2)
    """

    # Prepare the data for modeling
    df = prepare_data(df, value=value, feature_names=feature_names, split_method=split_method, fraction=fraction, seed=seed)

    # Train the model using AutoML
    automl = train_model(df, value='value', variables=feature_names, time_budget=time_budget, metric=metric, task=task, seed=seed)

    # Gather model statistics for testing, training, and all data
    mod_stats = pd.concat([
        modStats(df, automl, set='testing'),
        modStats(df, automl, set='training'),
        modStats(df.assign(set="all"), automl, set='all')
    ])

    # Variables to be used in resampling
    variables_resample = [item for item in feature_names if item not in ['hour', 'weekday', 'day_julian', 'date_unix']]

    # Normalize the data
    df_dew = normalise(automl, df, feature_names=feature_names, variables=variables_resample, n_samples=n_samples, n_cores=n_cores, seed=seed)

    # Initialize the dataframe for rolling window results
    dfr = pd.DataFrame(index=df_dew.index)
    df['date_d'] = pd.to_datetime(df['date']).dt.date
    date_max = pd.to_datetime(df['date_d'].max() - pd.DateOffset(days=window_days - 1))
    date_min = pd.to_datetime(df['date_d'].min() + pd.DateOffset(days=window_days - 1))

    # Apply the rolling window approach
    for i, ds in enumerate(pd.to_datetime(df['date_d'][df['date_d'] <= date_max.date()]).unique()[::rollingevery]):
        dfa = df[df['date_d'] >= ds.date()]
        dfa = dfa[dfa['date_d'] <= (dfa['date_d'].min() + pd.DateOffset(days=window_days)).date()]
        dfar = normalise(automl=automl, df=dfa, feature_names=feature_names, variables=variables_resample, n_samples=n_samples, n_cores=n_cores, seed=seed)
        dfar.rename(columns={'Normalised':'Rolling_'+str(i)},inplace=True)

        # Concatenate the results
        dfr = pd.concat([dfr, dfar['Rolling_'+str(i)]], axis=1)

    # Calculate the mean and standard deviation for the rolling window
    df_dew['EMI_mean_' + str(window_days)] = np.mean(dfr.iloc[:, 1:], axis=1)
    df_dew['EMI_std_' + str(window_days)] = np.std(dfr.iloc[:, 1:], axis=1)

    # Calculate the short-term and seasonal components
    df_dew['MET_short'] = df_dew['Observed'] - df_dew['EMI_mean_' + str(window_days)]
    df_dew['MET_season'] = df_dew['EMI_mean_' + str(window_days)] - df_dew['Normalised']

    return df_dew, mod_stats

def met_decom(df, value, feature_names, split_method='random', time_budget=60, metric='r2',
              estimator_list=["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"], task='regression',
              n_samples=300, fraction=0.75, seed=7654321, importance_ascending=False, n_cores=-1):
    """
    Decomposes a time series into different components using machine learning models with feature importance ranking.

    This function prepares the data, trains a machine learning model using AutoML, and decomposes the time series data
    into various components. The decomposition is based on the feature importance ranking and their contributions to the
    target variable. It returns the decomposed data and model statistics.

    Parameters:
        df (pd.DataFrame): Input dataframe containing the time series data.
        value (str): Column name of the target variable.
        feature_names (list of str): List of feature column names.
        split_method (str, optional): Method to split the data ('random' or other methods). Default is 'random'.
        time_budget (int, optional): Time budget for the AutoML training in seconds. Default is 60.
        metric (str, optional): Metric to evaluate the model ('r2', 'mae', etc.). Default is 'r2'.
        estimator_list (list of str, optional): List of estimators to be used in AutoML. Default is ["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"].
        task (str, optional): Task type ('regression' or 'classification'). Default is 'regression'.
        n_samples (int, optional): Number of samples for normalization. Default is 300.
        fraction (float, optional): Fraction of data to be used for training. Default is 0.75.
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
        >>> df_dewwc, mod_stats = met_decom(df, value, feature_names)
    """

    # Prepare the data for modeling
    df = prepare_data(df, value=value, feature_names=feature_names, split_method=split_method, fraction=fraction, seed=seed)

    # Train the model using AutoML
    automl = train_model(df, value='value', variables=feature_names, time_budget=time_budget, metric=metric, task=task, seed=seed)

    # Gather model statistics for testing, training, and all data
    mod_stats = pd.concat([
        modStats(df, automl, set='testing'),
        modStats(df, automl, set='training'),
        modStats(df.assign(set="all"), automl, set='all')
    ])

    # Determine feature importances and sort them
    var_names = feature_names
    automlfi = pd.DataFrame(data={'feature_importances': automl.feature_importances_},
                            index=automl.feature_names_in_).sort_values('feature_importances', ascending=importance_ascending)

    # Initialize the dataframe for decomposed components
    df_deww = df[['date', 'value']].set_index('date').rename(columns={'value': 'Observed'})
    MET_list = ['all'] + [item for item in automlfi.index if item not in ['hour', 'weekday', 'day_julian', 'date_unix']]

    # Decompose the time series by excluding different features based on their importance
    for var_to_exclude in MET_list:
        var_names = list(set(var_names) - set([var_to_exclude]))
        df_dew_temp = normalise(automl, df, feature_names=feature_names, variables=var_names, n_samples=n_samples, n_cores=n_cores, seed=seed)
        df_deww[var_to_exclude] = df_dew_temp['Normalised']

    # Adjust the decomposed components to create weather-independent values
    df_dewwc = df_deww.copy()
    for i, param in enumerate(MET_list):
        if (i > 0) & (i < len(MET_list)):
            df_dewwc[param] = df_deww[param] - df_deww[MET_list[i - 1]]

    return df_dewwc, mod_stats


def rolling_dew(df, value, feature_names, variables_resample, split_method='random', time_budget=60, metric='r2',
                estimator_list=["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"], task='regression',
                n_samples=300, window_days=14, rollingevery=2, fraction=0.75, seed=7654321, n_cores=-1):
    """
    Applies a rolling window approach to decompose the time series into different components using machine learning models.

    This function prepares the data, trains a machine learning model using AutoML, and applies a rolling window approach
    to decompose the time series data into various components. The decomposition is based on the contribution of different
    features to the target variable over rolling windows. It returns the decomposed data and model statistics.

    Parameters:
        df (pd.DataFrame): Input dataframe containing the time series data.
        value (str): Column name of the target variable.
        feature_names (list of str): List of feature column names.
        split_method (str, optional): Method to split the data ('random' or other methods). Default is 'random'.
        time_budget (int, optional): Time budget for the AutoML training in seconds. Default is 60.
        metric (str, optional): Metric to evaluate the model ('r2', 'mae', etc.). Default is 'r2'.
        estimator_list (list of str, optional): List of estimators to be used in AutoML. Default is ["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"].
        task (str, optional): Task type ('regression' or 'classification'). Default is 'regression'.
        variables_resample (list of str): List of sampled feature names for normalization.
        n_samples (int, optional): Number of samples for normalization. Default is 300.
        window_days (int, optional): Number of days for the rolling window. Default is 14.
        rollingevery (int, optional): Rolling interval in days. Default is 2.
        fraction (float, optional): Fraction of data to be used for training. Default is 0.75.
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

    # Prepare the data
    df = prepare_data(df, value=value, feature_names=feature_names, split_method=split_method, fraction=fraction, seed=seed)

    # Train the model using AutoML
    automl = train_model(df, value='value', variables=feature_names, time_budget=time_budget, metric=metric, task=task, seed=seed)

    # Collect model statistics
    mod_stats = pd.concat([
        modStats(df, automl, set='testing'),
        modStats(df, automl, set='training'),
        modStats(df.assign(set="all"), automl, set='all')
    ])

    # Create an initial dataframe to store observed values
    dfr = pd.DataFrame(index=df['date'], data={'Observed': list(df['value'])})
    df['date_d'] = df['date'].dt.date

    # Define the rolling window range
    date_max = df['date_d'].max() - pd.DateOffset(days=window_days - 1)
    date_min = df['date_d'].min() + pd.DateOffset(days=window_days - 1)

    # Iterate over the rolling windows
    for i, ds in enumerate(pd.to_datetime(df['date_d'][df['date_d'] <= date_max.date()]).unique()[::rollingevery]):
        dfa = df[df['date_d'] >= ds.date()]
        dfa = dfa[dfa['date_d'] <= (dfa['date_d'].min() + pd.DateOffset(days=window_days)).date()]

        # Normalize the data within the rolling window
        dfar = normalise(automl=automl, df=dfa, feature_names=feature_names, variables=variables_resample,
                         n_samples=n_samples, n_cores=n_cores, seed=seed)
        dfar.rename(columns={'Normalised':'Rolling_'+str(i)},inplace=True)

        # Concatenate the results
        dfr = pd.concat([dfr, dfar['Rolling_'+str(i)]], axis=1)

    return dfr, mod_stats

def do_all(df, value, feature_names, variables_resample, split_method='random', time_budget=60, metric='r2',
           estimator_list=["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"], task='regression',
           n_samples=300, fraction=0.75, seed=7654321, n_cores=-1):
    """
    Conducts data preparation, model training, and normalization, returning the transformed dataset and model statistics.

    This function performs the entire pipeline from data preparation to model training and normalization using
    specified parameters and returns the transformed dataset along with model statistics.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the dataset.
        value (str): Name of the target variable.
        feature_names (list of str): List of feature names.
        variables_resample (list of str): List of variables for normalization.
        split_method (str, optional): Method for splitting data ('random' or 'time_series'). Default is 'random'.
        time_budget (int, optional): Maximum time allowed for training models, in seconds. Default is 60.
        metric (str, optional): Evaluation metric for model performance ('r2', 'mae', etc.). Default is 'r2'.
        estimator_list (list of str, optional): List of estimator names to be used in training. Default is ["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"].
        task (str, optional): Task type ('regression' or 'classification'). Default is 'regression'.
        n_samples (int, optional): Number of samples for normalization. Default is 300.
        fraction (float, optional): Fraction of the dataset to be used for training. Default is 0.75.
        seed (int, optional): Seed for random operations. Default is 7654321.
        n_cores (int, optional): Number of CPU cores to be used for normalization (-1 for all available cores). Default is -1.

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

    # Prepare the data
    df = prepare_data(df, value=value, feature_names=feature_names, split_method=split_method, fraction=fraction, seed=seed)

    # Train the model using AutoML
    automl = train_model(df, value='value', variables=feature_names, time_budget=time_budget, metric=metric, task=task, seed=seed)

    # Collect model statistics
    mod_stats = pd.concat([
        modStats(df, automl, set='testing'),
        modStats(df, automl, set='training'),
        modStats(df.assign(set="all"), automl, set='all')
    ])

    # Normalize the data
    df_dew = normalise(automl, df, feature_names=feature_names, variables=variables_resample, n_samples=n_samples,aggregate=True, n_cores=n_cores, seed=seed)

    return df_dew, mod_stats

def do_all_unc(df, value, feature_names, variables_resample, split_method='random', time_budget=60, metric='r2',
               estimator_list=["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"], task='regression',
               n_models=10, confidence_level=0.95, n_samples=300, fraction=0.75, seed=7654321, n_cores=-1):
    """
    Performs uncertainty quantification by training multiple models with different random seeds and calculates statistical metrics.

    This function performs the entire pipeline from data preparation to model training and normalization, with an added step
    to quantify uncertainty by training multiple models using different random seeds. It returns a dataframe containing observed
    values, mean, standard deviation, median, confidence bounds, and weighted values, as well as a dataframe with model statistics.

    Parameters:
        df (pd.DataFrame): Input dataframe containing the time series data.
        value (str): Column name of the target variable.
        feature_names (list of str): List of feature column names.
        split_method (str, optional): Method to split the data ('random' or other methods). Default is 'random'.
        time_budget (int, optional): Time budget for the AutoML training. Default is 60.
        metric (str, optional): Metric to evaluate the model ('r2', 'mae', etc.). Default is 'r2'.
        estimator_list (list of str, optional): List of estimators to be used in AutoML. Default is ["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"].
        task (str, optional): Task type ('regression' or 'classification'). Default is 'regression'.
        n_models (int, optional): Number of models to train for uncertainty quantification. Default is 10.
        confidence_level (float, optional): Confidence level for the uncertainty bounds. Default is 0.95.
        variables_resample (list of str): List of sampled feature names for normalization (optional).
        n_samples (int, optional): Number of samples for normalization. Default is 300.
        fraction (float, optional): Fraction of data to be used for training. Default is 0.75.
        seed (int, optional): Random seed for reproducibility. Default is 7654321.
        n_cores (int, optional): Number of cores to be used (-1 for all available cores). Default is -1.

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
    df_dew = None
    mod_stats = None

    for i in random_seeds:
        df_dew0, mod_stats0 = do_all(df=df, value=value,
                                     feature_names=feature_names,
                                     split_method=split_method, time_budget=time_budget,
                                     variables_resample=variables_resample,
                                     n_samples=n_samples, fraction=fraction, seed=i, n_cores=n_cores)
        df_dew0.rename(columns={'Normalised':'Normalised_'+str(i)},inplace=True)
        df_dew0=df_dew0[['Observed','Normalised_'+str(i)]]
        if df_dew is None:
            df_dew = df_dew0
        else:
            df_dew = pd.concat([df_dew, df_dew0.iloc[:,1]], axis=1)

        mod_stats0['seed'] = i
        if mod_stats is None:
            mod_stats = mod_stats0
        else:
            mod_stats = pd.concat([mod_stats, mod_stats0], ignore_index=True)

    df_dew['mean'] = df_dew.iloc[:, 1:n_models+1].mean(axis=1)
    df_dew['std'] = df_dew.iloc[:, 1:n_models+1].std(axis=1)
    df_dew['median'] = df_dew.iloc[:, 1:n_models+1].median(axis=1)
    df_dew['lower_bound'] = df_dew.iloc[:, 1:n_models+1].quantile((1 - confidence_level) / 2, axis=1)
    df_dew['upper_bound'] = df_dew.iloc[:, 1:n_models+1].quantile(1 - (1 - confidence_level) / 2, axis=1)

    test_stats = mod_stats[mod_stats['set'] == 'testing']
    test_stats['R2'] = test_stats['R2'].replace([np.inf, -np.inf], np.nan)
    normalized_R2 = (test_stats['R2'] - test_stats['R2'].min()) / (test_stats['R2'].max() - test_stats['R2'].min())
    weighted_R2 = normalized_R2 / normalized_R2.sum()

    df_dew1 = df_dew.copy()
    df_dew1.iloc[:, 1:n_models+1] = df_dew.iloc[:, 1:n_models+1].values * weighted_R2.values
    df_dew['weighted'] = df_dew1.iloc[:, 1:n_models+1].sum(axis=1)

    return df_dew, mod_stats


def prepare_data(df, value, feature_names, prepared=False, na_rm=True, split_method='random', replace=False, fraction=0.75, seed=7654321):
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
          .pipe(check_data, value=value,feature_names=feature_names)
          .pipe(impute_values, na_rm=na_rm)
          .pipe(add_date_variables, replace=replace)
          .pipe(split_into_sets, split_method=split_method, fraction=fraction, seed=seed)
          .reset_index(drop=True))

    return df

def check_data(df, value, feature_names):
    """
    Validates and preprocesses the input DataFrame for subsequent analysis or modeling.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be checked.
    value : str
        The name of the target variable (column) to be used in the analysis.
    feature_names : list of str
        A list of feature names to be included in the analysis. If empty, all columns are used.

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
    - If `feature_names` is provided, only those columns (along with 'date' and the target column) are selected.
    """
    # Check if the target variable is in the DataFrame
    if value not in df.columns:
        raise ValueError("`value` is not within input data frame.")

    # Check if the date is in the index or columns
    if isinstance(df.index, pd.DatetimeIndex):
        date_in_index = True
    elif 'date' in df.columns:
        date_in_index = False
    else:
        raise ValueError("No datetime information found in index or 'date' column.")

    # Select features and the target variable
    if feature_names:
        selected_columns = list(set(feature_names) & set(df.columns))
    else:
        selected_columns = df.columns.tolist()

    # Ensure date and value columns are included
    if not date_in_index:
        selected_columns = selected_columns + ['date']
    selected_columns.append(value)

    # Select only the necessary columns
    df = df[selected_columns]

    # If the date is in the index, reset the index to a column for processing
    if date_in_index:
        df = df.reset_index()
        df = df.rename(columns={'index': 'date'})

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
        #df['weekday']=df['weekday'].astype("category")
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

def train_model(df, value, variables, time_budget=60, metric='r2',
                estimator_list=["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"],
                task='regression', seed=7654321, verbose=True):
    """
    Trains a machine learning model using the provided dataset and parameters.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the dataset.
        value (str): Name of the target variable.
        variables (list of str): List of feature variables.

    Keyword Parameters:
        time_budget (int, optional): Total running time in seconds. Default is 60.
        metric (str, optional): Primary metric for regression. Default is 'r2'.
        estimator_list (list, optional): List of ML learners. Default is ["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"].
        task (str, optional): Task type. Default is 'regression'.
        seed (int, optional): Random seed. Default is 7654321.
        verbose (bool, optional): Whether to print progress messages. Default is True.

    Returns:
        AutoML: Trained AutoML model object.

    Raises:
        ValueError: If `variables` contains duplicates or if any `variables` are not present in the DataFrame.
    """

    # Check for duplicate variables
    if len(set(variables)) != len(variables):
        raise ValueError("`variables` contains duplicate elements.")

    # Check if all variables are in the DataFrame
    if not all([var in df.columns for var in variables]):
        raise ValueError("`variables` given are not within input data frame.")

    # Filter DataFrame to include only the training set and relevant columns
    df = df.loc[df['set'] == 'training', [value] + variables]

    # AutoML settings
    automl_settings = {
        "time_budget": time_budget,  # Total running time in seconds
        "metric": metric,  # Primary metrics for regression can be chosen from: ['mae','mse','r2','rmse','mape']
        "estimator_list": estimator_list,  # List of ML learners
        "task": task,  # Task type
        "seed": seed,  # Random seed
        "verbose": verbose  # Verbose output
    }

    # Initialize and train AutoML model
    automl = AutoML()
    automl.fit(X_train=df[variables], y_train=df[value], **automl_settings)

    return automl


def normalise_worker(index, automl, df, variables, replace, n_samples, n_cores, seed, verbose):
    """
    Worker function for parallel normalization of data.

    Parameters:
        index (int): Index of the worker.
        automl (AutoML): Trained AutoML model.
        df (pd.DataFrame): Input DataFrame containing the dataset.
        variables (list of str): List of feature variables.
        replace (bool): Whether to sample with replacement.
        n_samples (int): Number of samples to normalize.
        n_cores (int): Number of CPU cores to use.
        seed (int): Random seed.
        verbose (bool): Whether to print progress messages.

    Returns:
        pd.DataFrame: DataFrame containing normalized predictions.
    """

    # Print progress message every fifth prediction
    if verbose and index % 5 == 0:
        # Calculate and format the progress percentage
        message_percent = round((index / n_samples) * 100, 2)
        message_percent = "{:.1f} %".format(message_percent)
        print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
              ": Predicting", index, "of", n_samples, "times (", message_percent, ")...")

    # Randomly sample observations
    np.random.seed(seed)
    n_rows = df.shape[0]
    index_rows = np.random.choice(range(n_rows), size=n_rows, replace=replace)

    # If variables are not provided, select all columns except 'date_unix'
    if variables is None:
        variables = list(set(df.columns) - {'date_unix'})

    # Create a sampled DataFrame
    df[variables] = df[variables].iloc[index_rows].reset_index(drop=True)

    # Use the model to predict
    value_predict = automl.predict(df)

    # Build DataFrame of predictions
    predictions = pd.DataFrame({
        'date': df['date'],
        'Observed': df['value'],
        'Normalised': value_predict
    })
    predictions['Seed']=seed

    return predictions


def normalise(automl, df, feature_names,variables, n_samples=300, replace=True,
                  aggregate=True, seed=7654321, n_cores=None,  verbose=True):
    """
    Normalizes the dataset using the trained model.

    Parameters:
        automl (object): Trained AutoML model.
        df (DataFrame): Input DataFrame containing the dataset.
        feature_names (list): List of feature names.

    Keyword Parameters:
        variables (list, optional): List of feature variables. Default is None.
        n_samples (int, optional): Number of samples to normalize. Default is 300.
        replace (bool, optional): Whether to replace existing data. Default is True.
        aggregate (bool, optional): Whether to aggregate results. Default is True.
        seed (int, optional): Random seed. Default is 7654321.
        n_cores (int, optional): Number of CPU cores to use. Default is None.
        verbose (bool, optional): Whether to print progress messages. Default is False.

    Returns:
        DataFrame: DataFrame containing normalized predictions.
    """

    # Default logic for cpu cores
    n_cores = n_cores if n_cores is not None else -1

    # Use all variables except the trend term
    if variables is None:
        #variables = automl.model.estimator.feature_name_
        variables = feature_names
        variables.remove('date_unix')

    # Sample the time series
    if verbose:
        print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), ": Sampling and predicting",
              n_samples, "times...")

    # If no samples are passed
    np.random.seed(seed)
    random_seeds = np.random.choice(np.arange(1000001), size=n_samples, replace=False)

    if n_samples == 0:
        df = pd.DataFrame()
    else:
        # Perform normalization using parallel processing
        df_result = pd.concat(Parallel(n_jobs=n_cores)(delayed(normalise_worker)(
            index=i, automl=automl, df=df, variables=variables, replace=replace,
            n_cores=n_cores, n_samples=n_samples, seed=random_seeds[i],
            verbose=False) for i in range(n_samples)), axis=0)
    # Aggregate results if needed
    if aggregate:
        df_result = df_result[['date','Observed','Normalised']].pivot_table(index='date', aggfunc='mean')[['Observed','Normalised']]
    else:
        # Pivot table to reshape 'Normalised' values by 'Seed' and set 'date' as index
        normalized_pivot = df_result.pivot_table(index='date', columns='Seed', values='Normalised')

        # Select and drop duplicate rows based on 'date', keeping only 'Observed' column
        observed_unique = df_result[['date', 'Observed']].drop_duplicates().set_index('date')

        # Concatenate the pivoted 'Normalised' values and unique 'Observed' values
        df_result = pd.concat([observed_unique, normalized_pivot], axis=1)

    return df_result


def modStats(df,automl,set=set,statistic=["n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA","R2"]):
    """
    Calculates statistics for model evaluation based on provided data.

    Parameters:
        df (DataFrame): Input DataFrame containing the dataset.
        set (str): Set type for which statistics are calculated ('training', 'testing', or 'all').
        statistic (list): List of statistics to calculate.

    Returns:
        DataFrame: DataFrame containing calculated statistics.
    """
    df=df[df['set']==set]
    df.loc[:,'value_predict']=automl.predict(df)
    df=Stats(df, mod="value_predict", obs="value",statistic=statistic).assign(set=set)
    return df

def Stats(df, mod="mod", obs="obs",
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
        res["p_value"] = r(df, mod, obs)[1]
    if "COE" in statistic:
        res["COE"] = COE(df, mod, obs)
    if "IOA" in statistic:
        res["IOA"] = IOA(df, mod, obs)
    if "R2" in statistic:
        res["R2"] = R2(df, mod, obs)

    results = {'n':res['n'], 'FAC2':res['FAC2'], 'MB':res['MB'], 'MGE':res['MGE'], 'NMB':res['NMB'],
               'NMGE':res['NMGE'],'RMSE':res['RMSE'], 'r':res['r'],'p_value':res['p_value'],
               'COE':res['COE'], 'IOA':res['IOA'], 'R2':res['R2']}

    results = pd.DataFrame([results])

    return results

## number of valid readings
def n(x, mod="mod", obs="obs"):
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
def FAC2(x, mod="mod", obs="obs"):
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
def MB(x, mod="mod", obs="obs"):
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
def MGE(x, mod="mod", obs="obs"):
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
def NMB(x, mod="mod", obs="obs"):
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
def NMGE(x, mod="mod", obs="obs"):
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
def RMSE(x, mod="mod", obs="obs"):
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
def r(x, mod="mod", obs="obs"):
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
def COE(x, mod="mod", obs="obs"):
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
def IOA(x, mod="mod", obs="obs"):
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
def R2(x, mod="mod", obs="obs"):
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
