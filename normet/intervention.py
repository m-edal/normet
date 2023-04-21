import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from typing import List
from operator import add
from toolz import reduce, partial
from scipy.optimize import fmin_slsqp
from joblib import Parallel, delayed


def SCM_parallel(data: pd.DataFrame, pollutant, intervention_date, treatment_target, control_targets = None, n_core = -1):
    control_pool = data["Code"].unique()

    synthetic_all = pd.concat(Parallel(n_jobs=n_core)(delayed(SCM)(
        data=data, pollutant=pollutant,intervention_date=intervention_date,treatment_target=Code,
        control_targets=control_targets) for Code in control_pool))
    return synthetic_all


def SCM(data: pd.DataFrame, pollutant, intervention_date,treatment_target, control_targets = None) -> np.array:
    if control_targets is None:
        control_targets=list(data['Code'].unique())
        control_targets.remove(treatment_target)

    inverted = (data.query(f"date<{intervention_date}")
                .pivot(index='Code', columns="date")[pollutant]
                .T)

    y = inverted[treatment_target].values # treated
    X = inverted[control_targets].values # donor pool

    weights = get_w(X, y)
    synthetic = (data.query(f"(Code=={control_targets})")
                 .pivot(index='date', columns="Code")[pollutant]
                 .values.dot(weights))
    data = (data
            .query(f"Code=={treatment_target}")[["Code", "date", pollutant]]
            .assign(Synthetic=synthetic))
    data['Effects']=data[pollutant]-data['Synthetic']

    return data

def pre_treatment_error(df,intervention_date):
    pre_treat_error = (df.query(f"date<{intervention_date}")["Effects"]) ** 2
    return pre_treat_error.mean()

def loss_w(W, X, y) -> float:
    return np.sqrt(np.mean((y - X.dot(W))**2))

def get_w(X, y):

    w_start = [1/X.shape[1]]*X.shape[1]

    weights = fmin_slsqp(partial(loss_w, X=X, y=y),
                         np.array(w_start),
                         f_eqcons=lambda x: np.sum(x) - 1,
                         bounds=[(0.0, 1.0)]*len(w_start),
                         disp=False)
    return weights
