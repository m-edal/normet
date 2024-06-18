import pandas as pd
import numpy as np

def cpd_rupture(df,col_name='Normalised',window=12, n=5,model="l2"):
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
    """
    import ruptures as rpt
    # Convert to numpy array
    values = np.array(df[col_name])

    # Perform changepoint detection
    model = rpt.Window(width=window, model="l2").fit(values)  # "l1", "rbf", "linear", "normal", "ar"
    result = model.predict(n_bkps=n)
    result = [x-1 for x in list(result)][:-1]

    # Convert changepoint indices to dates
    dates = df.iloc[result,:].index

    return dates



def cpd_cumsum(df,col_name='Normalised',threshold_mean= 10, threshold_std=3000):
    """
    Detects change points in a time series using cumulative sums method.

    Parameters:
        df (DataFrame): Input DataFrame containing the time series data.
        col_name (str, optional): Name of the column containing the time series data. Default is 'Normalised'.
        threshold_mean (float, optional): Threshold for mean change detection. Default is 10.
        threshold_std (float, optional): Threshold for standard deviation change detection. Default is 3000.

    Returns:
        DatetimeIndex: Datetime indices of detected change points.
    """
    data=df[col_name]
    mean_reference_value = np.mean(data)
    std_reference_value = np.std(data)

    S_mean = [0]
    S_std = [0]
    change_points = []

    for t in range(1, len(data)):
        S_mean.append(S_mean[t - 1] + (data[t] - mean_reference_value))
        S_std.append(S_std[t - 1] + ((abs(data[t] - mean_reference_value) - std_reference_value)))
        #pdb.set_trace()

        D_mean = abs(S_mean[t])
        D_std = abs(S_std[t])

        if D_mean > threshold_mean: #or D_std > threshold_std:
            #pdb.set_trace()
            change_points.append(t)
            S_mean[t] = 0
            S_std[t] = 0

    result = [x-1 for x in list(change_points)][:-1]
    dates = df.iloc[result,:].index

    return dates
