import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from random import sample
from scipy import stats
from flaml import AutoML
automl = AutoML()
from joblib import Parallel, delayed
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

def rolling_dew(df,value=None, window_days=30, feature_names=None, split_method = 'random',time_budget=60,metric= 'r2',
                  estimator_list=["lgbm", "rf","xgboost","extra_tree","xgb_limitdepth"],task='regression',
                  variables_sample=None, n_samples=300,fraction=0.75, seed=7654321, n_cores=-1):
    df=prepare_data(df, value=value, split_method = split_method,fraction=fraction,seed=seed)
    automl=train_model(df,variables=feature_names,
                time_budget= time_budget,  metric= metric, task= task, seed= seed);
    mod_stats=(pd.concat([modStats(df,set='testing'),
                modStats(df,set='training'),
                modStats(df.assign(set="all"),set='all')]))
    dfr=pd.DataFrame(index=df['date'],data={'Observed':list(df['value'])})
    for i in range(len(df[(df['date']>=df['date'].min())&(df['date']<=df['date'].max()-timedelta(days=window_days))])):
        dfa=df[(df['date']>=list(df['date'])[i])&(df['date']<=list(df['date'])[i+1] + timedelta(days=window_days))]
        dfar=normalise(automl=automl,df=dfa,
            feature_names=feature_names, variables= variables_sample,
        n_samples=n_samples, n_cores=n_cores, seed=seed)
        dfr=pd.concat([dfr,dfar.iloc[:,1]],axis=1)
    return dfr, mod_stats

def do_all_unc(df, value=None,feature_names=None, split_method = 'random',time_budget=60,metric= 'r2',
                estimator_list=["lgbm", "rf","xgboost","extra_tree","xgb_limitdepth"],task='regression',
                n_models=10, confidence_level=0.95, variables_sample=None, n_samples=300, fraction=0.75, seed=7654321, n_cores=-1):
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
    df=prepare_data(df, value=value, split_method = split_method,fraction=fraction,seed=seed)
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

def prepare_data(df, value='value', na_rm=False,split_method = 'random' ,replace=False, fraction=0.75,seed=7654321):

    # Check
    if value not in df.columns:
        raise ValueError("`value` is not within input data frame.")

    df = (df.rename(columns={value: "value"})
        .pipe(check_data, prepared=False)
        .pipe(impute_values, na_rm=na_rm)
        .pipe(add_date_variables, replace=replace)
        .pipe(split_into_sets, split_method = split_method,fraction=fraction,seed=seed)
        .reset_index(drop=True))

    return df

def add_date_variables(df, replace):

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
    # Remove missing values
    if na_rm:
        df = df.dropna(subset=['value'])
    # Numeric variables
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    # Character and categorical variables
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col].fillna(df[col].mode()[0],inplace=True)

    return df

def split_into_sets(df, split_method, fraction,seed):
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
    predictions = pd.DataFrame({'date': df['date'], 'Observed':df['value'],'Deweathered': value_predict})

    return predictions

def normalise(automl, df, feature_names,variables=None, n_samples=300, replace=True,
                  aggregate=True, seed=7654321, n_cores=None,  verbose=False):

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
    df=df[['Observed','Deweathered']].rename(columns={'Deweathered':'Deweathered_'+str(seed)})
    return df

def model_predict(automl, df=None):
    x = automl.predict(df)
    return x

def modStats(df,set=set,statistic=["n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA","R2"]):
    df=df[df['set']==set]
    df.loc[:,'value_predict']=automl.predict(df)
    df=Stats(df, mod="value_predict", obs="value",statistic=statistic).assign(set=set)
    return df

def Stats(df, mod="mod", obs="obs",
             statistic = ["n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA","R2"]):
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
    x = x[[mod, obs]].dropna()
    res = x.shape[0]
    return res

## fraction within a factor of two
def FAC2(x, mod="mod", obs="obs"):
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
    x = x[[mod, obs]].dropna()
    res = np.mean(x[mod] - x[obs])
    return res

## mean gross error
def MGE(x, mod="mod", obs="obs"):
    x = x[[mod, obs]].dropna()
    res = np.mean(np.abs(x[mod] - x[obs]))
    return res

## normalised mean bias
def NMB(x, mod="mod", obs="obs"):
    x = x[[mod, obs]].dropna()
    res = np.sum(x[mod] - x[obs]) / np.sum(x[obs])
    return res

## normalised mean gross error
def NMGE(x, mod="mod", obs="obs"):
    x = x[[mod, obs]].dropna()
    res = np.sum(np.abs(x[mod] - x[obs])) / np.sum(x[obs])
    return res

## root mean square error
def RMSE(x, mod="mod", obs="obs"):
    x = x[[mod, obs]].dropna()
    res = np.sqrt(np.mean((x[mod] - x[obs]) ** 2))
    return res

## correlation coefficient
def r(x, mod="mod", obs="obs"):
    x = x[[mod, obs]].dropna()
    res = stats.pearsonr(x[mod], x[obs])
    return res

## Coefficient of Efficiency
def COE(x, mod="mod", obs="obs"):
    x = x[[mod, obs]].dropna()
    res = 1 - np.sum(np.abs(x[mod] - x[obs])) / np.sum(np.abs(x[obs] - np.mean(x[obs])))
    return res

## Index of Agreement
def IOA(x, mod="mod", obs="obs"):
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
    x = x[[mod, obs]].dropna()
    X = sm.add_constant(x[obs])
    y=x[mod]
    model = sm.OLS(y, X).fit()
    res = model.rsquared
    return res
