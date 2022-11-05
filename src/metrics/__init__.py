import numpy as np
import pandas as pd
from scipy import stats
from src.metrics.helpers import get_bootstrap_samples
from scipy.stats import mannwhitneyu, ttest_ind_from_stats


def check_first_type_error(pilot: np.ndarray=None,
                           control: np.ndarray=None,
                           n_iter=10000,
                           alpha=0.05,
                           metric=None):
    """
    Calcs a first type error on 2 numpy arrays (pilot/prepilot, prepilot/history, etc)
    Parameters:
    --------------
    pilot: np.ndarray (default = None)
        numpy ndarray containing values for pilot group
    control: np.ndarray (default = None)
        numpy ndarray containing values for control group
    n_iter: int (default = 10000)
        Number of bootstrap iterations
    alpha: float (default = 0.05)
        Probability of rejecting the null hypothesis when it is true
    metric: str (default = None)
        Name of a business metric (margin/rto/etc)
    ----------------
    Returns:
    alpha_array: python tuple
        Tuple (alpha_empirical_mw, alpha_empirical_tt) calculated values (I type error for MW and TT tests)
    """
    counter_mw = 0
    counter_tt = 0
    if metric:
        print(pilot['plant'])
        pilot_element = [plant for plant in pilot['plant'].unique()]
        control_element = [plant for plant in control['plant'].unique()]
        print(pilot_element)
        print(control_element)

    mann_res = list()
    ttest_res = list()

    for i in range(n_iter):
        if metric:
            A = np.random.choice(pilot_element, len(pilot_element), replace=True)
            B = np.random.choice(control_element, len(control_element), replace=True)

            bs_pilot = pilot[pilot['plant'].isin(A)][metric].values
            bs_control = control[control['plant'].isin(B)][metric].values
        else:
            bs_pilot = get_bootstrap_samples(pilot)
            bs_control = get_bootstrap_samples(control)

        pval_mw = mannwhitneyu(bs_pilot.reshape(-1, 1),
                               bs_control.reshape(-1, 1)).pvalue
        mann_res.append(pval_mw)
        pval_tt = ttest_ind_from_stats(bs_pilot.mean(), bs_pilot.std(), bs_pilot.shape[0], bs_control.mean(),
                                       bs_control.std(), bs_control.shape[0], equal_var=False).pvalue
        ttest_res.append(pval_tt)
        if pval_mw < alpha:
            counter_mw += 1
        if pval_tt < alpha:
            counter_tt += 1
    alpha_empirical_mw = counter_mw / n_iter
    alpha_empirical_tt = counter_tt / n_iter
    alpha_array = (alpha_empirical_mw, alpha_empirical_tt)

    return alpha_array


def check_second_type_error(pilot: np.ndarray=None,
                            control: np.ndarray=None,
                            n_iter=10000,
                            beta=0.05,
                            effects=[0.01, 0.03, 0.05, 0.1],
                            metric=None):
    """
    Calcs a first type error on 2 numpy arrays (pilot/prepilot, prepilot/history, etc)
    Parameters:
    ----------------
    pilot: np.ndarray (default = None)
        numpy ndarray containing values for pilot group
    control: np.ndarray (default = None)
        numpy ndarray containing values for control group
    n_iter: int (default = 10000)
        Number of bootstrap iterations
    beta: float (default = 0.05)
        Probability of rejecting the alternative hypothesis when it is true
    effects: python list (default = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2])
        Possible values for an effect size (we calc II type error for several possible effect sizes)
    metric: str (default = None)
        Name of a business metric (margin/rto/etc)
    --------------------
    Returns:
    betta_array: python tuple
        Tuple (effects, alpha_empirical_mw, alpha_empirical_tt) calculated values (II type errors list for MW and TT tests)
    """
    if metric:
        pilot_element = [plant for plant in pilot['plant'].unique()]
        control_element = [plant for plant in control['plant'].unique()]
        print(pilot_element)
        print(control_element)

    mw_errors = []
    tt_errors = []
    for effect in effects:
        ttest_res = []
        mann_res = []

        for i in range(n_iter):
            if metric:
                A = np.random.choice(pilot_element, len(pilot_element), replace=True)
                B = np.random.choice(control_element, len(control_element), replace=True)

                bs_pilot = pilot[pilot['plant'].isin(A)][metric].values
                bs_control = control[control['plant'].isin(B)][metric].values
            else:
                bs_pilot = get_bootstrap_samples(pilot)
                bs_control = get_bootstrap_samples(control)

            mean_pilot = np.mean(bs_pilot)
            std_pilot = np.std(bs_pilot)
            noise = np.random.normal(
                mean_pilot*effect, std_pilot / 10, size=len(bs_control))

            bs_control = bs_control + noise
            ttest = stats.ttest_ind(bs_pilot, bs_control, equal_var=False)
            mann = stats.mannwhitneyu(bs_pilot, bs_control)

            ttest_res.append(ttest[1])
            mann_res.append(mann[1])

        beta_empirical_tt = (np.array(ttest_res) > beta).mean()
        beta_empirical_mw = (np.array(mann_res) > beta).mean()
        mw_errors.append(beta_empirical_mw)
        tt_errors.append(beta_empirical_tt)

    result_tuple = (effects, mw_errors, tt_errors)
    return result_tuple
