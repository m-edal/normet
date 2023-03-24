#import pandas as pd
#import numpy as np
#import multiprocessing as mp
#from datetime import datetime
#from joblib import Parallel, delayed

def normet_normalise_worker(index, automl, df, variables, replace, n_samples,n_cores, verbose):
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
    index_rows = np.random.choice(range(n_rows), size=n_rows, replace=replace)

    # Transform data frame to include sampled variables
    if variables is None:
        variables = list(set(df.columns) - {'date_unix'})
    # Transform data frame to include sampled variables
    df[variables] = df[variables].iloc[index_rows].reset_index(drop=True)

    # Use model to predict
    value_predict = normet_predict(automl, df)

    # Build data frame of predictions
    predictions = pd.DataFrame({'date': df['date'], 'Observed':df['value'],'Deweathered': value_predict})
    predictions=predictions[['Observed','Deweathered']]

    return predictions

def normet_normalise(automl, df, feature_names,variables=None, n_samples=300, replace=True,
                  aggregate=True, n_cores=None, verbose=False):

    df = normet_check_data(df, prepared=True)
    # Default logic for cpu cores
    n_cores = n_cores if n_cores is not None else mp.cpu_count()

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
    if n_samples == 0:
        df = pd.DataFrame()
    else:
        df = pd.concat(Parallel(n_jobs=n_cores)(delayed(normet_normalise_worker)(
            index=i,automl=automl,df=df,
            variables=variables,replace=replace,n_cores=n_cores,
            n_samples=n_samples,
            verbose=verbose) for i in range(n_samples)), axis=0).pivot_table(index='date',aggfunc='mean')
    df=df[['Observed','Deweathered']]
    return df
