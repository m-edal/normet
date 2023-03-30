import pandas as pd
import numpy as np

def rpt_breakpoints(df,window=12, n=5,model="l2"):
    import ruptures as rpt
    # Convert to numpy array
    values = np.array(df['Deweathered'])

    # Perform changepoint detection
    model = rpt.Window(width=window, model="l2").fit(values)
    result = model.predict(n_bkps=n)
    result = [x-1 for x in list(result)][:-1]

    # Convert changepoint indices to dates
    dates = pd.to_datetime(df.iloc[result,:].index)

    return dates
