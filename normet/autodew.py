import pandas as pd
import numpy as np
from datetime import datetime
from random import sample
from scipy import stats
from flaml import AutoML
automl = AutoML()
from joblib import Parallel, delayed
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


def ts_decom(df, value=None,feature_names=None, split_method = 'random',time_budget=60,metric= 'r2',
                  estimator_list=["lgbm", "rf","xgboost","extra_tree","xgb_limitdepth"],task='regression',
                  n_samples=300,fraction=0.75, seed=7654321, n_cores=-1):
    """
    Decomposes a time series into different components using machine learning models.

    Parameters:
        df (pd.DataFrame): Input dataframe containing the time series data.
        value (str): Column name of the target variable.
        feature_names (list of str): List of feature column names.
        split_method (str): Method to split the data ('random' or other methods).
        time_budget (int): Time budget for the AutoML training.
        metric (str): Metric to evaluate the model ('r2', 'mae', etc.).
        estimator_list (list of str): List of estimators to be used in AutoML.
        task (str): Task type ('regression' or 'classification').
        n_samples (int): Number of samples for normalisation.
        fraction (float): Fraction of data to be used for training.
        seed (int): Random seed for reproducibility.
        n_cores (int): Number of cores to be used (-1 for all available cores).

    Returns:
        df_dewc (pd.DataFrame): Dataframe with decomposed components.
        mod_stats (pd.DataFrame): Dataframe with model statistics.
    """
    df=prepare_data(df, value=value, feature_names=feature_names, split_method = split_method,fraction=fraction,seed=seed)
    automl=train_model(df,variables=feature_names,
                time_budget= time_budget,  metric= metric, task= task, seed= seed);
    mod_stats=(pd.concat([modStats(df,set='testing'),
                modStats(df,set='training'),
                modStats(df.assign(set="all"),set='all')]))
    var_names=feature_names
    df_dew=df[['date','value']].set_index('date').rename(columns={'value':'Observed'})
    for var_to_exclude in ['all','date_unix','day_julian','weekday','hour']:
        var_names = list(set(var_names) - set([var_to_exclude]))
        df_dew_temp = normalise(automl, df,
            feature_names=feature_names,
            variables=var_names,
            n_samples=n_samples,
            n_cores=n_cores,
            seed=seed)
        df_dew[var_to_exclude] = df_dew_temp.iloc[:, 1]

    df_dewc=df_dew.copy()
    df_dewc['hour']=df_dew['hour']-df_dew['weekday']
    df_dewc['weekday']=df_dew['weekday']-df_dew['day_julian']
    df_dewc['day_julian']=df_dew['day_julian']-df_dew['date_unix']
    df_dewc['date_unix']=df_dew['date_unix']-df_dew['all']+df_dew['hour'].mean()
    df_dewc['Deweathered']=df_dew['hour']
    return df_dewc, mod_stats


def met_rolling(df, value=None,feature_names=None, split_method = 'random',time_budget=60,metric= 'r2',
                  estimator_list=["lgbm", "rf","xgboost","extra_tree","xgb_limitdepth"],task='regression',
                  n_samples=300,window_days=15,rollingevery=2,fraction=0.75, seed=7654321, n_cores=-1):
    """
    Applies a rolling window approach to decompose the time series into different components using machine learning models.

    Parameters:
        df (pd.DataFrame): Input dataframe containing the time series data.
        value (str): Column name of the target variable.
        feature_names (list of str): List of feature column names.
        split_method (str): Method to split the data ('random' or other methods).
        time_budget (int): Time budget for the AutoML training.
        metric (str): Metric to evaluate the model ('r2', 'mae', etc.).
        estimator_list (list of str): List of estimators to be used in AutoML.
        task (str): Task type ('regression' or 'classification').
        n_samples (int): Number of samples for normalisation.
        window_days (int): Number of days for the rolling window.
        rollingevery (int): Rolling interval.
        fraction (float): Fraction of data to be used for training.
        seed (int): Random seed for reproducibility.
        n_cores (int): Number of cores to be used (-1 for all available cores).

    Returns:
        df_dew (pd.DataFrame): Dataframe with decomposed components.
        mod_stats (pd.DataFrame): Dataframe with model statistics.
    """
    df=prepare_data(df, value=value, feature_names=feature_names, split_method = split_method,fraction=fraction,seed=seed)
    automl=train_model(df,variables=feature_names,
                time_budget= time_budget,  metric= metric, task= task, seed= seed);
    mod_stats=(pd.concat([modStats(df,set='testing'),
                modStats(df,set='training'),
                modStats(df.assign(set="all"),set='all')]))
    variables_sample=[item for item in feature_names if item not in ['hour','weekday','day_julian','date_unix']]
    df_dew=normalise(automl, df,
                           feature_names = feature_names,
                          variables= variables_sample,
                          n_samples=n_samples, n_cores=n_cores, seed=seed)

    dfr=pd.DataFrame(index=df_dew.index)
    df['date_d']=df['date'].dt.date
    date_max=df['date_d'].max()-pd.DateOffset(days=window_days-1)
    date_min=df['date_d'].min()+pd.DateOffset(days=window_days-1)
    for i,ds in enumerate(df['date_d'][df['date_d']<=date_max].unique()[::rollingevery]):
        dfa=df[df['date_d']>=ds]
        dfa=dfa[dfa['date']<=dfa['date'].min()+pd.DateOffset(days=window_days)]
        dfar=normalise(automl=automl,df=dfa,
            feature_names=feature_names, variables= variables_sample,
            n_samples=n_samples, n_cores=n_cores, seed=seed)
        dfr=pd.concat([dfr,dfar.iloc[:,1]],axis=1)
    df_dew['EMI_mean_'+str(window_days)]=np.mean(dfr.iloc[:,1:],axis=1)
    df_dew['EMI_std_'+str(window_days)]=np.mean(dfr.iloc[:,1:],axis=1)
    df_dew['MET_short']=df_dew['Observed']-df_dew['EMI_mean_'+str(window_days)]
    df_dew['MET_season']=df_dew['EMI_mean_'+str(window_days)]-df_dew['Normalised_'+str(seed)]
    return df_dew, mod_stats

def met_decom(df,value=None,feature_names=None, split_method = 'random',time_budget=60,metric= 'r2',
                  estimator_list=["lgbm", "rf","xgboost","extra_tree","xgb_limitdepth"],task='regression',
                  n_samples=300,fraction=0.75, seed=7654321, importance_ascending=False, n_cores=-1):
    """
    Decomposes a time series into different components using machine learning models with feature importance ranking.

    Parameters:
        df (pd.DataFrame): Input dataframe containing the time series data.
        value (str): Column name of the target variable.
        feature_names (list of str): List of feature column names.
        split_method (str): Method to split the data ('random' or other methods).
        time_budget (int): Time budget for the AutoML training.
        metric (str): Metric to evaluate the model ('r2', 'mae', etc.).
        estimator_list (list of str): List of estimators to be used in AutoML.
        task (str): Task type ('regression' or 'classification').
        n_samples (int): Number of samples for normalisation.
        fraction (float): Fraction of data to be used for training.
        seed (int): Random seed for reproducibility.
        importance_ascending (bool): Sort order for feature importances.
        n_cores (int): Number of cores to be used (-1 for all available cores).

    Returns:
        df_dewwc (pd.DataFrame): Dataframe with decomposed components.
        mod_stats (pd.DataFrame): Dataframe with model statistics.
    """
    df=prepare_data(df, value=value, feature_names=feature_names, split_method = split_method,fraction=fraction,seed=seed)
    automl=train_model(df,variables=feature_names,
                time_budget= time_budget,metric= metric, task= task, seed= seed);
    mod_stats=(pd.concat([modStats(df,set='testing'),
                modStats(df,set='training'),
                modStats(df.assign(set="all"),set='all')]))
    var_names=feature_names
    automlfi=pd.DataFrame(data={'feature_importances':automl.feature_importances_},
                      index=automl.feature_names_in_).sort_values('feature_importances',ascending=importance_ascending)
    df_deww=df[['date','value']].set_index('date').rename(columns={'value':'Observed'})
    MET_list=['all']+[item for item in automlfi.index if item not in ['hour','weekday','day_julian','date_unix']]
    for var_to_exclude in MET_list:
        var_names = list(set(var_names) - set([var_to_exclude]))
        df_dew_temp = normalise(automl, df,
            feature_names=feature_names,
            variables=var_names,
            n_samples=n_samples,
            n_cores=n_cores,
            seed=seed)
        df_deww[var_to_exclude] = df_dew_temp.iloc[:, 1]
    df_dewwc=df_deww.copy()
    for i,param in enumerate(MET_list):
        if (i>0)&(i<len(MET_list)):
            df_dewwc[param]=df_deww[param]-df_deww[MET_list[i-1]]

    return df_dewwc, mod_stats

def rolling_dew(df,value=None, feature_names=None, split_method = 'random',time_budget=60,metric= 'r2',
                  estimator_list=["lgbm", "rf","xgboost","extra_tree","xgb_limitdepth"],task='regression',
                  variables_sample=None, n_samples=300,window_days=15, rollingevery=2,fraction=0.75, seed=7654321, n_cores=-1):
    """
    Applies a rolling window approach to decompose the time series into different components using machine learning models.

    Parameters:
        df (pd.DataFrame): Input dataframe containing the time series data.
        value (str): Column name of the target variable.
        feature_names (list of str): List of feature column names.
        split_method (str): Method to split the data ('random' or other methods).
        time_budget (int): Time budget for the AutoML training.
        metric (str): Metric to evaluate the model ('r2', 'mae', etc.).
        estimator_list (list of str): List of estimators to be used in AutoML.
        task (str): Task type ('regression' or 'classification').
        variables_sample (list of str): List of sampled feature names for normalisation (optional).
        n_samples (int): Number of samples for normalisation.
        window_days (int): Number of days for the rolling window.
        rollingevery (int): Rolling interval.
        fraction (float): Fraction of data to be used for training.
        seed (int): Random seed for reproducibility.
        n_cores (int): Number of cores to be used (-1 for all available cores).

    Returns:
        dfr (pd.DataFrame): Dataframe with rolling decomposed components.
        mod_stats (pd.DataFrame): Dataframe with model statistics.
    """

    # Prepare the data
    df=prepare_data(df, value=value, feature_names=feature_names,split_method = split_method,fraction=fraction,seed=seed)

    # Train the model using AutoML
    automl=train_model(df,variables=feature_names,
                time_budget= time_budget,  metric= metric, task= task, seed= seed);

    # Collect model statistics
    mod_stats=(pd.concat([modStats(df,set='testing'),
                modStats(df,set='training'),
                modStats(df.assign(set="all"),set='all')]))

    # Create an initial dataframe to store observed values
    dfr=pd.DataFrame(index=df['date'],data={'Observed':list(df['value'])})
    df['date_d']=df['date'].dt.date

    # Define the rolling window range
    date_max=df['date_d'].max()-pd.DateOffset(days=window_days-1)
    date_min=df['date_d'].min()+pd.DateOffset(days=window_days-1)

    # Iterate over the rolling windows
    for i,ds in enumerate(df['date_d'][df['date_d']<=date_max].unique()[::rollingevery]):
        dfa=df[df['date_d']>=ds]
        dfa=dfa[dfa['date']<=dfa['date'].min()+pd.DateOffset(days=window_days)]

        # Normalize the data within the rolling window
        dfar=normalise(automl=automl,df=dfa,
            feature_names=feature_names, variables= variables_sample,
            n_samples=n_samples, n_cores=n_cores, seed=seed)

        # Concatenate the results
        dfr=pd.concat([dfr,dfar.iloc[:,1]],axis=1)
    return dfr, mod_stats

def do_all_unc(df, value=None,feature_names=None, split_method = 'random',time_budget=60,metric= 'r2',
                estimator_list=["lgbm", "rf","xgboost","extra_tree","xgb_limitdepth"],task='regression',
                n_models=10, confidence_level=0.95, variables_sample=None, n_samples=300, fraction=0.75, seed=7654321, n_cores=-1):
    """
    Performs uncertainty quantification by training multiple models with different random seeds and calculates statistical metrics.

    Parameters:
        df (pd.DataFrame): Input dataframe containing the time series data.
        value (str): Column name of the target variable.
        feature_names (list of str): List of feature column names.
        split_method (str): Method to split the data ('random' or other methods).
        time_budget (int): Time budget for the AutoML training.
        metric (str): Metric to evaluate the model ('r2', 'mae', etc.).
        estimator_list (list of str): List of estimators to be used in AutoML.
        task (str): Task type ('regression' or 'classification').
        n_models (int): Number of models to train for uncertainty quantification.
        confidence_level (float): Confidence level for the uncertainty bounds.
        variables_sample (list of str): List of sampled feature names for normalisation (optional).
        n_samples (int): Number of samples for normalisation.
        fraction (float): Fraction of data to be used for training.
        seed (int): Random seed for reproducibility.
        n_cores (int): Number of cores to be used (-1 for all available cores).

    Returns:
        df_dew (pd.DataFrame): Dataframe with observed values, mean, standard deviation, median, lower and upper bounds, and weighted values.
        mod_stats (pd.DataFrame): Dataframe with model statistics.
    """
    np.random.seed(seed)
    random_seeds = np.random.choice(np.arange(1000001), size=n_models, replace=False)
    mod_stats=pd.DataFrame(columns=['n','FAC2','MB','MGE','NMB','NMGE','RMSE','r','p_value','COE',
                                   'IOA','R2','set','seed'])
    df_dew=df.set_index('date')[[value]].rename(columns={value:'Observed'})
    for i in random_seeds:
        df_dew0,mod_stats0=do_all(df=df, value=value,
                                 feature_names=feature_names,
                                 split_method = split_method,time_budget=time_budget,
                                 variables_sample=variables_sample,
                                 n_samples=n_samples,fraction=fraction,seed=i, n_cores=n_cores)
        df_dew=pd.concat([df_dew,df_dew0.iloc[:,1]],axis=1)
        mod_stats0['seed']=i
        mod_stats=pd.concat([mod_stats,mod_stats0])
    df_dew['mean']=df_dew.iloc[:,1:n_models+1].mean(axis=1)
    df_dew['std']=df_dew.iloc[:,1:n_models+1].std(axis=1)
    df_dew['median']=df_dew.iloc[:,1:n_models+1].median(axis=1)
    df_dew['lower_bound'] = df_dew.iloc[:,1:n_models+1].quantile((1 - confidence_level) / 2,axis=1)
    df_dew['upper_bound'] = df_dew.iloc[:,1:n_models+1].quantile(1 - (1 - confidence_level) / 2,axis=1)

    test_stats = mod_stats[mod_stats['set'] == 'testing']
    test_stats['R2']=test_stats['R2'].replace([np.inf, -np.inf], np.nan)
    normalized_R2 = (test_stats['R2'] - test_stats['R2'].min()) / (test_stats['R2'].max() - test_stats['R2'].min())
    weighted_R2 = normalized_R2 / normalized_R2.sum()

    df_dew1 = df_dew.copy()
    df_dew1.iloc[:, 1:n_models+1] = df_dew.iloc[:, 1:n_models+1].values * weighted_R2.values
    df_dew['weighted'] = df_dew1.iloc[:, 1:n_models+1].sum(axis=1)
    return df_dew, mod_stats

def do_all(df, value=None,feature_names=None, split_method = 'random',time_budget=60,metric= 'r2',
                  estimator_list=["lgbm", "rf","xgboost","extra_tree","xgb_limitdepth"],task='regression',
                  variables_sample=None, n_samples=300,fraction=0.75, seed=7654321, n_cores=-1):
    """
    Conducts data preparation, model training, and normalisation, returning the transformed dataset and model statistics.

    Parameters:
        df (DataFrame): Input DataFrame containing the dataset.
        value (str, optional): Name of the target variable. Default is None.
        feature_names (list, optional): List of feature names. Default is None.
        split_method (str, optional): Method for splitting data ('random' or 'time_series'). Default is 'random'.
        time_budget (int, optional): Maximum time allowed for training models, in seconds. Default is 60.
        metric (str, optional): Evaluation metric for model performance. Default is 'r2'.
        estimator_list (list, optional): List of estimator names to be used in training. Default is ["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"].
        task (str, optional): Task type ('regression' or 'classification'). Default is 'regression'.
        variables_sample (list, optional): List of variables for normalisation. Default is None.
        n_samples (int, optional): Number of samples for normalisation. Default is 300.
        fraction (float, optional): Fraction of the dataset to be used for training. Default is 0.75.
        seed (int, optional): Seed for random operations. Default is 7654321.
        n_cores (int, optional): Number of CPU cores to be used for normalisation. Default is -1 (use all available cores).

    Returns:
        tuple: Transformed dataset and model statistics DataFrame.
    """
    df=prepare_data(df, value=value, feature_names=feature_names,split_method = split_method,fraction=fraction,seed=seed)
    automl=train_model(df,variables=feature_names,
                time_budget= time_budget,  metric= metric, task= task, seed= seed);
    mod_stats=(pd.concat([modStats(df,set='testing'),
                modStats(df,set='training'),
                modStats(df.assign(set="all"),set='all')]))

    df_dew=normalise(automl, df,
                           feature_names = feature_names,
                          variables= variables_sample,
                          n_samples=n_samples, n_cores=n_cores, seed=seed)
    return df_dew, mod_stats

def prepare_data(df, value='value', feature_names=None, na_rm=True,split_method = 'random' ,replace=False, fraction=0.75,seed=7654321):
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
    # Check
    if value not in df.columns:
        raise ValueError("`value` is not within input data frame.")

    df=df[list(set(feature_names) & set(list(df.columns)))+['date',value]]
    df = (df.rename(columns={value: "value"})
        .pipe(check_data, prepared=False)
        .pipe(impute_values, na_rm=na_rm)
        .pipe(add_date_variables, replace=replace)
        .pipe(split_into_sets, split_method = split_method,fraction=fraction,seed=seed)
        .reset_index(drop=True))

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
        df[col].fillna(df[col].median(), inplace=True)
    # Character and categorical variables
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col].fillna(df[col].mode()[0],inplace=True)

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

def check_data(df, prepared):
    """
    Checks the integrity of the input DataFrame.

    Parameters:
        df (DataFrame): Input DataFrame containing the dataset.
        prepared (bool): Whether the DataFrame is already prepared.

    Returns:
        DataFrame: DataFrame with checked integrity.
    """
    if 'date' not in df.columns:
        raise ValueError("Input must contain a `date` variable.")
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        raise ValueError("`date` variable needs to be a parsed date (datetime64).")
    if df['date'].isnull().any():
        raise ValueError("`date` must not contain missing (NA) values.")

    if prepared:
        if 'set' not in df.columns:
            raise ValueError("Input must contain a `set` variable.")
        if not set(df['set'].unique()).issubset(set(['training', 'testing'])):
            raise ValueError("`set` can only take the values `training` and `testing`.")
        if "value" not in df.columns:
            raise ValueError("Input must contain a `value` variable.")
        if "date_unix" not in df.columns:
            raise ValueError("Input must contain a `date_unix` variable.")
    return df


def train_model(df, variables,
    time_budget= 60,  # total running time in seconds
    metric= 'r2',  # primary metrics for regression can be chosen from: ['mae','mse','r2','rmse','mape']
    estimator_list= ["lgbm", "rf","xgboost","extra_tree","xgb_limitdepth"],  # list of ML learners; we tune lightgbm in this example
    task= 'regression',  # task type
    seed= 7654321,    # random seed
    verbose = True
):
    """
    Trains a model using the provided dataset and Args.

    Parameters:
        df (DataFrame): Input DataFrame containing the dataset.
        variables (list): List of feature variables.

    Keyword Parameters:
        time_budget (int, optional): Total running time in seconds. Default is 60.
        metric (str, optional): Primary metric for regression. Default is 'r2'.
        estimator_list (list, optional): List of ML learners. Default is ["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth"].
        task (str, optional): Task type. Default is 'regression'.
        seed (int, optional): Random seed. Default is 7654321.
        verbose (bool, optional): Whether to print progress messages. Default is True.

    Returns:
        object: Trained model.
    """
    # Check arguments
    if len(set(variables)) != len(variables):
        raise ValueError("`variables` contains duplicate elements.")

    if not all([var in df.columns for var in variables]):
        raise ValueError("`variables` given are not within input data frame.")

    # Check input dataset
    df = check_data(df, prepared=True)

    # Filter and select input for modelling
    df = df.loc[df['set'] == 'training', ['value'] + variables]

    automl_settings = {
        "time_budget": time_budget,  # total running time in seconds
        "metric": metric,  # primary metrics for regression can be chosen from: ['mae','mse','r2','rmse','mape']
        "estimator_list": estimator_list,  # list of ML learners; we tune lightgbm in this example
        "task": task,  # task type
        "seed": seed,    # random seed
        "verbose": verbose
    }

    automl.fit(X_train=df[variables], y_train=df['value'],**automl_settings)

    return automl

def normalise_worker(index, automl, df, variables, replace, n_samples,n_cores, seed, verbose):
    """
    Worker function for parallel normalisation of data.

    Parameters:
        index (int): Index of the worker.
        automl (object): Trained AutoML model.
        df (DataFrame): Input DataFrame containing the dataset.
        variables (list): List of feature variables.
        replace (bool): Whether to replace existing data.
        n_samples (int): Number of samples to normalize.
        n_cores (int): Number of CPU cores to use.
        seed (int): Random seed.
        verbose (bool): Whether to print progress messages.

    Returns:
        DataFrame: DataFrame containing normalized predictions.
    """
    # Only every fifth prediction message
    if verbose and index % 5 == 0:
        # Calculate percent
        message_percent = round((index / n_samples) * 100, 2)
        # Always have 2 dp
        message_percent = "{:.1f} %".format(message_percent)
        # Print
        print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
              ": Predicting", index, "of", n_samples, "times (", message_precent, ")...")
    # Randomly sample observations
    n_rows = df.shape[0]
    np.random.seed(seed)
    index_rows = np.random.choice(range(n_rows), size=n_rows, replace=replace)

    # Transform data frame to include sampled variables
    if variables is None:
        variables = list(set(df.columns) - {'date_unix'})
    # Transform data frame to include sampled variables
    df[variables] = df[variables].iloc[index_rows].reset_index(drop=True)

    # Use model to predict
    value_predict = model_predict(automl, df)

    # Build data frame of predictions
    predictions = pd.DataFrame({'date': df['date'], 'Observed':df['value'],'Normalised': value_predict})

    return predictions

def normalise(automl, df, feature_names,variables=None, n_samples=300, replace=True,
                  aggregate=True, seed=7654321, n_cores=None,  verbose=False):
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
    df = check_data(df, prepared=True)
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
        df = pd.concat(Parallel(n_jobs=n_cores)(delayed(normalise_worker)(
            index=i,automl=automl,df=df,
            variables=variables,replace=replace,n_cores=n_cores,
            n_samples=n_samples,seed=random_seeds[i],
            verbose=verbose) for i in range(n_samples)), axis=0).pivot_table(index='date',aggfunc='mean')
    df=df[['Observed','Normalised']].rename(columns={'Normalised':'Normalised_'+str(seed)})
    return df

def model_predict(automl, df=None):
    """
    Predicts values using the trained model.

    Parameters:
        automl (object): Trained AutoML model.
        df (DataFrame, optional): DataFrame containing data to predict. Default is None.

    Returns:
        array: Predicted values.
    """
    x = automl.predict(df)
    return x

def modStats(df,set=set,statistic=["n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA","R2"]):
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
