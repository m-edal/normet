import pandas as pd
import numpy as np

def cpt_rupture(df,col_name='Deweathered',window=12, n=5,model="l2"):
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



def cpt_cumsum(df,col_name='Deweathered',threshold_mean= 10, threshold_std=3000):
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
