#require packages: import pandas as pd; import numpy as np; from random import sample; from scipy.stats import mode; from datetime import datetime

def normet_prepare_data(df, value='value', na_rm=False,split_method = 'random' ,replace=False, fraction=0.75):

    # Check
    if value not in df.columns:
        raise ValueError("`value` is not within input data frame.")

    df = (df.rename(columns={value: "value"})
        .pipe(normet_check_data, prepared=False)
        .pipe(impute_values, na_rm=na_rm)
        .pipe(add_date_variables, replace=replace)
        .pipe(split_into_sets, split_method = split_method,fraction=fraction)
        .reset_index(drop=True))

    return df

def add_date_variables(df, replace):

    if replace:
        # Will replace if variables exist
        df['date_unix'] = df['date'].astype(np.int64) // 10**9
        df['day_julian'] = pd.DatetimeIndex(df['date']).dayofyear
        df['weekday'] = pd.DatetimeIndex(df['date']).weekday + 1
        df['hour'] = pd.DatetimeIndex(df['date']).hour

    else:
         # Add variables if they do not exist
         # Add date variables
        if 'date_unix' not in df.columns:
            df['date_unix'] = df['date'].apply(lambda x: x.timestamp())
        if 'day_julian' not in df.columns:
            df['day_julian'] = df['date'].apply(lambda x: x.timetuple().tm_yday)

        # An internal package's function
        if 'weekday' not in df.columns:
            df['weekday'] = df['date'].apply(lambda x: x.weekday() + 1)

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
        df[col].fillna(mode(df[col])[0][0], inplace=True)

    return df

def split_into_sets(df, split_method, fraction):
    # Add row number
    df = df.reset_index().rename(columns={'index': 'rowid'})
    if (split_method == 'random'):
        # Sample to get training set
        df_training = df.sample(frac=fraction, random_state=42).reset_index(drop=True).assign(set="training")
        # Remove training set from input to get testing set
        df_testing = df[~df['rowid'].isin(df_training['rowid'])].assign(set="testing")
    if (split_method == 'time_series'):
        df_training = df.iloc[:int(fraction*df.shape[0]),:].reset_index(drop=True).assign(set="training")
        df_testing = df[~df['rowid'].isin(df_training['rowid'])].assign(set="testing")

    # Bind again
    df_split = pd.concat([df_training, df_testing], axis=0, ignore_index=True)
    #df_split = df_split[['date', 'value',  'date_unix', 'day_julian', 'weekday', 'hour','set']]
    df_split = df_split.sort_values(by='date').reset_index(drop=True)

    return df_split

def normet_check_data(df, prepared):

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
