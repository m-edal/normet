
#required packages: import pandas as pd; import numpu as np;  from scipy import stats
#𝑛: the number of complete pairs of data.
#FAC2: fraction of predictions within a factor of two.
#MB: the mean bias.
#MGE: the mean gross error.
#NMB: the normalised mean bias.
#NMGE: the normalised mean gross error.
#RMSE: the root mean squared error.
#r: the Pearson correlation coefficient.
#COE: the Coefficient of Efficiency. A perfect model has a COE = 1. A value of COE = 0.0 has a fundamental meaning. For negative values of COE, the model is less effective than the observed mean in predicting the variation in the observations.
#IOA: the Index of Agreement, which spans between -1 and +1 with values approaching +1 representing better model performance.

def normet_modStats(df,value=None,split_method = 'random',set='testing',fraction=0.75):
    df=normet_prepare_data(df, value=value, split_method = split_method,fraction=fraction)
    dft=df[df['set']==set]
    dft['value_predict']=automl.predict(dft)
    df=modStats(dft, mod="value_predict", obs="value")
    return df


def modStats(df, mod="mod", obs="obs",
             statistic=["n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA"],):
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
        res["p_Value"] = r(df, mod, obs)[1]
    if "COE" in statistic:
        res["COE"] = COE(df, mod, obs)
    if "IOA" in statistic:
        res["IOA"] = IOA(df, mod, obs)

    results = {'n':res['n'], 'FAC2':res['FAC2'], 'MB':res['MB'], 'MGE':res['MGE'], 'NMB':res['NMB'],
               'NMGE':res['NMGE'],
               'RMSE':res['RMSE'], 'r':res['r'],'p_Value':res['p_Value'], 'COE':res['COE'], 'IOA':res['IOA']}

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
# when SD=0; will return(NA)
def r(x, mod="mod", obs="obs"):
    x = x[[mod, obs]].dropna()
    res = stats.pearsonr(x[mod], x[obs])
    #return pd.DataFrame({"r": [res[0]], "P": [res[1]]})
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
