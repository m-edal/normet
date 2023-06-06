import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import cvxpy as cp
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

def scm_parallel(df, poll_col, date_col, code_col, control_pool, post_col, n_cores = -1):

    treatment_pool = df[code_col].unique()
    synthetic_all = pd.concat(Parallel(n_jobs=n_cores)(delayed(scm)(
                    df=df,
                    poll_col=poll_col,
                    date_col=date_col,
                    code_col=code_col,
                    treat_target=code,
                    control_pool=control_pool,
                    post_col=post_col) for code in treatment_pool))
    return synthetic_all


def scm(df, poll_col, date_col, code_col, treat_target, control_pool, post_col):

    x_pre_control = (df[(df[code_col]!=treat_target)&(df[code_col].isin(control_pool))]
                     .query(f"~{post_col}")
                     .pivot(date_col, code_col, poll_col)
                     .values)

    y_pre_treat_mean = (df
                        .query(f"~{post_col}")[df[code_col]==treat_target].groupby(date_col)
                        [poll_col]
                        .mean())

    #w = cp.Variable(x_pre_control.shape[1])
    #objective = cp.Minimize(cp.sum_squares(x_pre_control@w - y_pre_treat_mean.values))
    #constraints = [cp.sum(w) == 1, w >= 0]

    #problem = cp.Problem(objective, constraints)
    #problem.solve(verbose=False)

    #alpha = 1.0  # 岭回归的正则化参数
    param_grid = {'alpha': [i/10 for i in range(1, 101)]}  # 可以根据需要设置不同的 alpha 值
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=5)  # cv 表示交叉验证的折数
    grid_search.fit(x_pre_control, y_pre_treat_mean.values.reshape(-1, 1))  # 在训练集上拟合模型
    best_alpha = grid_search.best_params_['alpha']
    ridge_final = Ridge(alpha=best_alpha,fit_intercept=False)
    ridge_final.fit(x_pre_control, y_pre_treat_mean.values.reshape(-1, 1))  # 在训练集上拟合模型


    # 使用岭回归拟合合成对照组权重
    #ridge = Ridge(alpha=alpha, fit_intercept=False)
    #ridge.fit(x_pre_control, y_pre_treat_mean.values.reshape(-1, 1))
    w = ridge_final.coef_.flatten()

    #sc = (df[(df[code_col]!=treat_target)&(df[code_col].isin(control_pool))]
    #      .pivot(date_col, code_col, poll_col)
    #      .values) @ w.value
    sc = (df[(df[code_col]!=treat_target)&(df[code_col].isin(control_pool))]
      .pivot(date_col, code_col, poll_col)
      .values) @ w

    data=(df
            [df[code_col]==treat_target][[date_col, code_col, poll_col]]
            .assign(synthetic=sc)).set_index(date_col)
    data['effects']=data[poll_col]-data['synthetic']
    return data


def sdid(df, poll_col, date_col, code_col, treat_target, control_pool, post_col):

    # find the unit weights
    unit_weights = fit_unit_weights(df,
                                    poll_col=poll_col,
                                    date_col=date_col,
                                    code_col=code_col,
                                    treat_target=treat_target,
                                    control_pool=control_pool,
                                    post_col=post_col)

    # find the time weights
    time_weights = fit_time_weights(df,
                                    poll_col=poll_col,
                                    date_col=date_col,
                                    code_col=code_col,
                                    treat_target=treat_target,
                                    control_pool=control_pool,
                                    post_col=post_col)

    # join weights into DiD Data
    did_data = join_weights(df, unit_weights, time_weights,
                            date_col=date_col,
                            code_col=code_col,
                            treat_target=treat_target,
                            control_pool=control_pool,
                            post_col=post_col)

    df['treated']=df[code_col]==treat_target
    # run DiD
    formula = f"{poll_col} ~ {post_col}* treated"
    did_model = smf.wls(formula, data=did_data, weights=did_data["weights"]+1e-10).fit()

    return did_model.params[f"{post_col}:treated"]

def sdid_effects(df, poll_col, date_col, code_col, treat_target,control_pool, post_col):
    effects = {date: sdid(df[(~df[post_col])|(df[date_col]==date)],
                          poll_col=poll_col,
                          date_col=date_col,
                          code_col=code_col,
                          treat_target=treat_target,
                          control_pool=control_pool,
                          post_col=post_col)
               for date in list(df.query(f"{post_col}")[date_col].unique())}
    effects=pd.DataFrame(pd.Series(effects)).rename_axis(index=date_col).rename(columns={0:'effects'})
    effects[code_col]=treat_target
    return effects

def sdid_parallel(df, poll_col, date_col, code_col, control_pool, post_col,n_cores = -1):

    treatment_pool = df[code_col].unique()

    sdid_all = pd.concat(Parallel(n_jobs=n_cores)(delayed(sdid_effects)(
        df=df, poll_col=poll_col,date_col=date_col, code_col=code_col,treat_target=Code,
        control_pool=control_pool, post_col=post_col) for Code in treatment_pool))
    sdid_all.index.name=date_col
    sdid_all.rename(columns={0:'effects'},inplace=True)
    return sdid_all


def calculate_regularization(df, poll_col, date_col, code_col, treat_target,control_pool, post_col):

    n_treated_post = (df.query(f"{post_col}")[df[code_col]==treat_target].shape[0])

    first_diff_std = (df[(df[code_col]!=treat_target)&(df[code_col].isin(control_pool))]
                      .query(f"~{post_col}")
                      .sort_values(date_col)
                      .groupby(code_col)
                      [poll_col]
                      .diff()
                      .std())

    return n_treated_post**(1/4) * first_diff_std


def fit_unit_weights(df, poll_col, date_col, code_col, treat_target,control_pool, post_col):

    zeta = calculate_regularization(df, poll_col, date_col, code_col, treat_target,control_pool,post_col)
    pre_data = df.query(f"~{post_col}")

    # pivot the data to the (T_pre, N_co) matrix representation
    y_pre_control = (pre_data[(pre_data[code_col]!=treat_target)&(pre_data[code_col].isin(control_pool))]
                     .pivot(date_col, code_col, poll_col))

    # group treated units by time periods to have a (T_pre, 1) vector.
    y_pre_treat_mean = (pre_data[pre_data[code_col]==treat_target]
                        .groupby(date_col)
                        [poll_col]
                        .mean())

    # add a (T_pre, 1) column to the begining of the (T_pre, N_co) matrix to serve as intercept
    T_pre = y_pre_control.shape[0]
    X = np.concatenate([np.ones((T_pre, 1)), y_pre_control.values], axis=1)

    # estimate unit weights. Notice the L2 penalty using zeta
    w = cp.Variable(X.shape[1])
    objective = cp.Minimize(cp.sum_squares(X@w - y_pre_treat_mean.values) + T_pre*zeta**2 * cp.sum_squares(w[1:]))
    constraints = [cp.sum(w[1:]) == 1, w[1:] >= 0]

    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False)
    #alpha = T_pre * zeta**2  # 岭回归的正则化参数

    # 使用岭回归拟合单位权重
    #ridge = Ridge(alpha=alpha, fit_intercept=False)
    #ridge.fit(X, y_pre_treat_mean.values)
    #w = ridge.coef_

    # print("Intercept:", w.value[0])
    return pd.Series(w.value[1:], # remove intercept
                     name="unit_weights",
                     index=y_pre_control.columns)
    #return pd.Series(w[1:], name="unit_weights", index=y_pre_control.columns)



def fit_time_weights(df, poll_col, date_col, code_col, treat_target,control_pool, post_col):

        control = df[(df[code_col]!=treat_target)&(df[code_col].isin(control_pool))]

        # pivot the data to the (T_pre, N_co) matrix representation
        y_pre = (control
                 .query(f"~{post_col}")
                 .pivot(date_col, code_col, poll_col))

        # group post-treatment time period by units to have a (1, N_co) vector.
        y_post_mean = (control
                       .query(f"{post_col}")
                       .groupby(code_col)
                       [poll_col]
                       .mean()
                       .values)

        # add a (1, N_co) vector of 1 to the top of the matrix, to serve as the intercept.
        X = np.concatenate([np.ones((1, y_pre.shape[1])), y_pre.values], axis=0)

        # estimate time weights
        w = cp.Variable(X.shape[0])
        objective = cp.Minimize(cp.sum_squares(w@X - y_post_mean))
        constraints = [cp.sum(w[1:]) == 1, w[1:] >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=False)

        # print("Intercept: ", w.value[0])
        return pd.Series(w.value[1:], # remove intercept
                         name="time_weights",
                         index=y_pre.index)


def join_weights(df, unit_w, time_w, date_col, code_col, treat_target,control_pool, post_col):
    df['treated']=df[code_col]==treat_target

    return (
        df
        .set_index([date_col, code_col])
        .join(time_w)
        .join(unit_w)
        .reset_index()
        .fillna({time_w.name: df['after_treatment'].mean(),
                 unit_w.name: df['treated'].mean()})
        .assign(**{"weights": lambda d: (d[time_w.name]*d[unit_w.name]).round(10)})
        .astype({'treated':int, 'after_treatment':int}))
