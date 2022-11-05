import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from scipy.stats import mannwhitneyu, zscore, ttest_ind, ttest_ind_from_stats
from statsmodels.stats.proportion import proportions_ztest


from .helpers import calc_stratified_std
from src.metrics.plots import effect_plot_ptl, effect_plot_px

def calc_effect_bootstrap(
        a=None,
        b=None,
        name=None,
        title=None,
        n_iter=10000,
        level=[2.5, 97.5],
        output_path=None):
    """
    Calcs confidence interval with a selected level (default equals to 95%)
    --------------
    Parameters:
    a: numpy array (default = None)
        Control group values array
    b: numpy array (default = None)
        Experiment group values array
    name: str (default = None)
        Image File prefix (*.png)
    title: str (default = None)
        Image caption
    n_iter: int (default = 10000)
        Number of effect samples
    level: python list (default = [2.5, 97.5])
        Left and right confidence level intervals
    metric: str (default = None)
        metric on which we measure an effect distribution
    output_path: str (default = None)
        Path to dump reports and images
    ---------------
    Returns:
    report: python dictionary
        contains left_border, mean_value, right_border of an effect size
    """

    res = []
    for i in range(n_iter):

        a_sample = np.random.choice(a=a,
                                      size=len(a),
                                      replace=True)
        b_sample = np.random.choice(a=b,
                                        size=len(b),
                                        replace=True)

        effect = np.mean(b_sample) / np.mean(a_sample) - 1
        res.append(effect)

    res = np.array(res)
    res = res * 100
    left_border, right_border = np.percentile(res, level)

    #effect_plot_ptl(res, left_border, right_border, title, output_path, name)
    effect_plot_px(res, left_border, right_border, title, name)

    report = {'left_border': left_border,
              'right_border': right_border,
              'mean_value': np.mean(res)}

    return report

def calc_confidence_interval_pilot_prepilot(
        pilot=None,
        control=None,
        prepilot=None,
        precontrol=None,
        name=None,
        title=None,
        element='country_code',
        n_iter=10000,
        level=[2.5, 97.5],
        metric=None,
        output_path=None):
    """
    Calcs confidence interval with a selected level (default equals to 95%)
    --------------
    Parameters:
    pilot: pandas DataFrame (default = None)
        Experiment plants data for a pilot period
    control: pandas DataFrame (default = None)
        Control plants data for a pilot period
    prepilot: pandas DataFrame (default = None)
        Experiment plants data for a PREpilot period
    precontrol: pandas DataFrame (default = None)
        Control plants data for a PREpilot period
    name: str (default = None)
        Image File prefix (*.png)
    title: str (default = None)
        Image caption
    element: str (default = 'Plant')
        element name
    n_iter: int (default = 10000)
        Number of effect samples
    level: python list (default = [2.5, 97.5])
        Left and right confidence level intervals
    metric: str (default = None)
        metric on which we measure an effect distribution
    output_path: str (default = None)
        Path to dump reports and images
    ---------------
    Returns:
    report: python dictionary
        contains left_border, mean_value, right_border of an effect size
    """
    pilot_element = [plant for plant in pilot[element].unique()]
    control_element = [plant for plant in control[element].unique()]
    pilot = pilot.reset_index()
    prepilot = prepilot.reset_index()
    control = control.reset_index()
    precontrol = precontrol.reset_index()
    pilot.index = pilot[element]
    prepilot.index = prepilot[element]
    control.index = control[element]
    precontrol.index = precontrol[element]

    res = []
    for i in range(n_iter):
        # sampling from pilot and control elements
        #A = np.random.choice(pilot_element, len(pilot_element), replace=True)
        #B = np.random.choice(control_element, len(control_element), replace=True)

        #pilot_vect = pilot.loc[A, metric].values
        #prepilot_vect = prepilot.loc[A, metric].values

        #control_vect = control.loc[B, metric].values
        #precontrol_vect = precontrol.loc[B, metric].values

        pilot_vect = np.random.choice(a=pilot[metric].values,
                                      size=len(pilot),
                                      replace=True)
        control_vect = np.random.choice(a=control[metric].values,
                                        size=len(control),
                                        replace=True)
        prepilot_vect = np.random.choice(a=prepilot[metric].values,
                                         size=len(prepilot),
                                         replace=True)
        precontrol_vect = np.random.choice(a=precontrol[metric].values,
                                           size=len(precontrol),
                                           replace=True)

        effect = np.mean(pilot_vect) / np.mean(control_vect) - 1
        preeffect = np.mean(prepilot_vect) / np.mean(precontrol_vect) - 1
        effect = effect - preeffect
        res.append(effect)

    res = np.array(res)
    res = res * 100
    left_border, right_border = np.percentile(res, level)

    figure = plt.gca()
    plt.figure(figsize=(20, 10))
    plt.hist(res, bins=200, color='c', edgecolor='k', alpha=0.65)
    plt.axvline(np.mean(res), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(left_border, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(right_border, color='r', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()

    plt.text(np.mean(res) * 1.1, max_ylim * 0.9, 'Mean: {:.2f}% '.format(np.mean(res)))
    plt.text(left_border * 1.1, max_ylim * 0.9, '{:.2f}% '.format(left_border))
    plt.text(right_border * 1.1, max_ylim * 0.9, '{:.2f}% '.format(right_border))
    y_axis = figure.axes.get_yaxis()
    y_axis.set_visible(False)
    plt.title(title)
    plt.savefig(output_path + '/' + name + '_conf_level.png')

    report = {'left_border': left_border,
              'right_border': right_border,
              'mean_value': np.mean(res)}

    return report

def calc_confidence_interval(
        dat_check, title, n_iter=10000, level=[2.5, 97.5]):
    dat_pilot = dat_check[['pilot', 'pilot_check']].drop_duplicates()
    elements = ["{:02d}".format(x) for x in range(0, dat_pilot.shape[0])]
    dat_pilot.index = dat_pilot.pilot.astype(str) + '_' + elements
    group_A = dat_pilot.index.values

    dat_control = dat_check[['control', 'control_check']]
    elements = ["{:02d}".format(x) for x in range(0, dat_control.shape[0])]
    dat_control.index = dat_control.control.astype(str) + '_' + elements
    group_B = dat_control.index.values

    # i = 1
    res = []
    for i in tqdm(range(n_iter)):
        print(i)
        checkA = np.random.choice(group_A, len(group_A), replace=True)
        checkB = np.random.choice(group_B, len(group_B), replace=True)

        pilot_vect_A = dat_pilot.loc[checkA, 'pilot_check'].values
        pilot_vect_B = dat_control.loc[checkB, 'control_check'].values

        effect = np.mean(pilot_vect_A) / np.mean(pilot_vect_B) - 1

        res.append(effect)

    res = np.array(res) * 100
    left_border, right_border = np.percentile(res, level)

    figure = plt.gca()
    plt.figure(figsize=(20, 10))
    plt.hist(res, bins=100, color='c', edgecolor='k', alpha=0.65)
    plt.axvline(np.mean(res), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(left_border, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(right_border, color='r', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()

    plt.text(np.mean(res) * 1.1, max_ylim * 0.9, 'Mean: {:.2f}% '.format(np.mean(res)))
    plt.text(left_border * 1.1, max_ylim * 0.9, '{:.2f}% '.format(left_border))
    plt.text(right_border * 1.1, max_ylim * 0.9, '{:.2f}% '.format(right_border))
    y_axis = figure.axes.get_yaxis()
    y_axis.set_visible(False)
    plt.title('Confidence Interval. Effect Distribution')

    plt.savefig(title + '_conf_level.png')
    plt.show()

    print(
        "Нижняя граница: {:.2f}% ".format(left_border),
        "Среднее значение: {:.2f}% ".format(np.mean(res)),
        "Верхняя граница: {:.2f}% ".format(right_border),
    )

    return

def calc_confidence_interval_pilot_year_to_year(
        pilot, control, prepilot, precontrol,
        name, title,
        element='plant',
        n_iter=10000, level=[2.5, 97.5]):
    # Берем только сущности в препилоте, которые были и в пилоте и в препилоте
    pilot_element = np.intersect1d(pilot[element].values, prepilot[element].values)
    prepilot = prepilot[prepilot[element].isin(pilot_element)]
    pilot = pilot[pilot[element].isin(pilot_element)]

    # Берем только сущности в преконтроле, которые были и в контроле и в преконтроле
    control_element = np.intersect1d(control[element].values, precontrol[element].values)
    precontrol = precontrol[precontrol[element].isin(control_element)]
    control = control[control[element].isin(control_element)]

    pilot = pilot.reset_index()
    prepilot = prepilot.reset_index()
    control = control.reset_index()
    precontrol = precontrol.reset_index()

    pilot.index = pilot[element]
    prepilot.index = prepilot[element]
    control.index = control[element]
    precontrol.index = precontrol[element]

    # i = 1
    res1 = []
    res2 = []
    for i in range(n_iter):
        print(i)
        # Фиксируем рандомный контроль и пилот
        A = np.random.choice(pilot_element, len(pilot_element), replace=True)
        B = np.random.choice(control_element, len(control_element), replace=True)

        pilot_vect = pilot.loc[A, 'rto'].values
        prepilot_vect = prepilot.loc[A, 'rto'].values

        control_vect = control.loc[B, 'rto'].values
        precontrol_vect = precontrol.loc[B, 'rto'].values

        effect = np.mean(pilot_vect) / np.mean(control_vect) - 1
        preeffect = np.mean(prepilot_vect) / np.mean(precontrol_vect) - 1
        effect1 = effect - preeffect

        # KNN
        effect = np.mean(pilot_vect) / np.mean(prepilot_vect)
        preeffect = np.mean(control_vect) / np.mean(precontrol_vect)
        effect2 = effect - preeffect

        res1.append(effect1)
        res2.append(effect2)

    res = res2
    res = np.array(res) * 100
    left_border, right_border = np.percentile(res, level)

    figure = plt.gca()
    plt.figure(figsize=(20, 10))
    plt.hist(res, bins=100, color='c', edgecolor='k', alpha=0.65)
    plt.axvline(np.mean(res), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(left_border, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(right_border, color='r', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()

    plt.text(np.mean(res) * 1.1, max_ylim * 0.9, 'Mean: {:.2f}% '.format(np.mean(res)))
    plt.text(left_border * 1.1, max_ylim * 0.9, '{:.2f}% '.format(left_border))
    plt.text(right_border * 1.1, max_ylim * 0.9, '{:.2f}% '.format(right_border))
    y_axis = figure.axes.get_yaxis()
    y_axis.set_visible(False)
    plt.title(title + 'Year to Year')

    plt.savefig(name + '_conf_level_1.png')
    plt.show()

    print(
        "Нижняя граница: {:.2f}% ".format(left_border),
        "Среднее значение: {:.2f}% ".format(np.mean(res)),
        "Верхняя граница: {:.2f}% ".format(right_border),
    )

    res = res1
    res = np.array(res) * 100
    left_border, right_border = np.percentile(res, level)

    figure = plt.gca()
    plt.figure(figsize=(20, 10))
    plt.hist(res, bins=100, color='c', edgecolor='k', alpha=0.65)
    plt.axvline(np.mean(res), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(left_border, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(right_border, color='r', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()

    plt.text(np.mean(res) * 1.1, max_ylim * 0.9, 'Mean: {:.2f}% '.format(np.mean(res)))
    plt.text(left_border * 1.1, max_ylim * 0.9, '{:.2f}% '.format(left_border))
    plt.text(right_border * 1.1, max_ylim * 0.9, '{:.2f}% '.format(right_border))
    y_axis = figure.axes.get_yaxis()
    y_axis.set_visible(False)
    plt.title(title + 'Diff of Diff')

    plt.savefig(name + '_conf_level_2.png')
    plt.show()

    print(
        "Нижняя граница: {:.2f}% ".format(left_border),
        "Среднее значение: {:.2f}% ".format(np.mean(res)),
        "Верхняя граница: {:.2f}% ".format(right_border),
    )
    return


def calc_confidence_interval_pilot_prepilot_values(
        pilot, control, prepilot, precontrol,
        name, title,
        element='plant',
        n_iter=10000, level=[2.5, 97.5]):
    # Берем только сущности в препилоте, которые были и в пилоте и в препилоте
    pilot_element = np.intersect1d(pilot[element].values, prepilot[element].values)
    prepilot = prepilot[prepilot[element].isin(pilot_element)]
    pilot = pilot[pilot[element].isin(pilot_element)]

    # Берем только сущности в преконтроле, которые были и в контроле и в преконтроле
    control_element = np.intersect1d(control[element].values, precontrol[element].values)
    precontrol = precontrol[precontrol[element].isin(control_element)]
    control = control[control[element].isin(control_element)]

    pilot = pilot.reset_index()
    prepilot = prepilot.reset_index()
    control = control.reset_index()
    precontrol = precontrol.reset_index()

    pilot.index = pilot[element]
    prepilot.index = prepilot[element]
    control.index = control[element]
    precontrol.index = precontrol[element]

    # i = 1
    res = []
    for i in range(n_iter):
        print(i)
        pilot_vect = np.random.choice(pilot['rto'].values, pilot.shape[0], replace=True)
        control_vect = np.random.choice(control['rto'].values, control.shape[0], replace=True)
        prepilot_vect = np.random.choice(prepilot['rto'].values, prepilot.shape[0], replace=True)
        precontrol_vect = np.random.choice(precontrol['rto'].values, precontrol.shape[0], replace=True)

        effect = np.mean(pilot_vect) / np.mean(control_vect) - 1
        preeffect = np.mean(prepilot_vect) / np.mean(precontrol_vect) - 1

        effect = effect - preeffect
        res.append(effect)

    res = np.array(res) * 100
    left_border, right_border = np.percentile(res, level)

    figure = plt.gca()
    plt.figure(figsize=(20, 10))
    plt.hist(res, bins=100, color='c', edgecolor='k', alpha=0.65)
    plt.axvline(np.mean(res), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(left_border, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(right_border, color='r', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()

    plt.text(np.mean(res) * 1.1, max_ylim * 0.9, 'Mean: {:.2f}% '.format(np.mean(res)))
    plt.text(left_border * 1.1, max_ylim * 0.9, '{:.2f}% '.format(left_border))
    plt.text(right_border * 1.1, max_ylim * 0.9, '{:.2f}% '.format(right_border))
    y_axis = figure.axes.get_yaxis()
    y_axis.set_visible(False)
    plt.title(title)

    plt.savefig(name + '_conf_level.png')
    plt.show()

    print(
        "Нижняя граница: {:.2f}% ".format(left_border),
        "Среднее значение: {:.2f}% ".format(np.mean(res)),
        "Верхняя граница: {:.2f}% ".format(right_border),
    )
    return

def first_type_error(df: pd.DataFrame = None,
                           strat_column: str = 'platform',
                           measured_metric: str = 'ARPU_inapp',
                           ctr: bool = True,
                           n_iter: int = 1000,
                           alpha: float = 0.05,
                           post_stratification: bool = False):
    values_a = df[df['experiment_group']=='A'][measured_metric].values
    values_b = df[df['experiment_group']=='B'][measured_metric].values
    p_values = []

    for i in range(n_iter):
        a = np.random.choice(a=values_a,
                             replace=True,
                             #size=sample_size
                             size=len(values_a)
                            )
        b = np.random.choice(a=values_b,
                             replace=True,
                             #size=sample_size
                             size=len(values_b)
                            )
        if ctr:
            count = np.array([len(a[a == 1]), len(b[b == 1])])
            size = np.array([len(a), len(b)])
            p_value = proportions_ztest(count, size)[1]
        else:
            if post_stratification:
                a = df[df['experiment_group']=='A']
                b = df[df['experiment_group']=='B']
                a = a.iloc[np.random.randint(len(a),
                                             #size=sample_size)
                                             size=len(a))
                          ]
                b = b.iloc[np.random.randint(len(b),
                                             #size=sample_size)
                                             size=len(b))
                          ]
                A_mean, B_mean = np.mean(a[measured_metric].values), np.mean(b[measured_metric].values)
                #print(a)
                #print(b)
                A_std = calc_stratified_std(df=a,
                                  strat_column=strat_column,
                                  experiment_group='A',
                                  metric=measured_metric)
                B_std = calc_stratified_std(df=b,
                                  strat_column=strat_column,
                                  experiment_group='B',
                                  metric=measured_metric)
            else:
                A_mean, B_mean, A_std, B_std = np.mean(a), np.mean(b), a.std(), b.std()
            p_value = ttest_ind_from_stats(mean1=A_mean, std1=A_std, nobs1=len(a),
                                          mean2=B_mean, std2=B_std, nobs2=len(b),
                                          equal_var=False)[1]
        p_values.append(p_value)
    p_values = np.array(p_values)
    #print(p_values)
    I_type_error = len(p_values[p_values < alpha]) / len(p_values)
    return I_type_error


def second_type_error(df: pd.DataFrame = None,
                      strat_column: str = 'platform',
                      measured_metric: str = 'ARPU_inapp',
                      ctr: bool = True,
                      n_iter: int = 1000,
                      alpha: float = 0.05,
                      post_stratification: bool = False,
                      effect: float = 0.03):
    values_a = df[df['experiment_group'] == 'A'][measured_metric].values
    values_b = df[df['experiment_group'] == 'B'][measured_metric].values
    p_values = []

    for i in range(n_iter):
        a = np.random.choice(a=values_a,
                             replace=True,
                             #size=sample_size
                             size=len(values_a)
                             )
        b = np.random.choice(a=values_b,
                             replace=True,
                             #size=sample_size
                             size=len(values_b)
                             )
        if ctr:
            num_ones_a = int(len(a[a == 1]) * effect)
            count = np.array([len(a[a == 1]), len(b[b == 1]) + num_ones_a])
            size = np.array([len(a), len(b)])
            p_value = proportions_ztest(count, size)[1]
        else:

            effect_values = np.random.normal(np.mean(a) * effect,
                                             np.std(a) / 10,
                                             len(b)) + b
            # print(np.sum(a)/np.sum(effect_values))

            if post_stratification:
                a = df[df['experiment_group'] == 'A']
                b = df[df['experiment_group'] == 'B']
                a = a.iloc[np.random.randint(len(a),
                                             size=len(a))
                    # size=len(a))
                ]
                b = b.iloc[np.random.randint(len(b),
                                             size=len(b))
                    # size=len(b))
                ]
                A_mean = np.mean(a[measured_metric].values)
                # print(a)
                # print(b)
                A_std = calc_stratified_std(df=a,
                                            strat_column=strat_column,
                                            experiment_group='A',
                                            metric=measured_metric)

                effect_values = np.random.normal(A_mean * effect,
                                                 A_std / 10,
                                                 len(b))

                b[measured_metric] = b[measured_metric] + effect_values
                B_mean = np.mean(b[measured_metric].values)
                B_std = calc_stratified_std(df=b,
                                            strat_column=strat_column,
                                            experiment_group='B',
                                            metric=measured_metric)

            else:
                A_mean, B_mean, A_std, B_std = np.mean(a), np.mean(effect_values), a.std(), effect_values.std()
            p_value = ttest_ind_from_stats(mean1=A_mean, std1=A_std, nobs1=len(a),
                                           mean2=B_mean, std2=B_std, nobs2=len(effect_values),
                                           equal_var=False)[1]
        p_values.append(p_value)
    p_values = np.array(p_values)
    II_type_error = len(p_values[p_values > alpha]) / len(p_values)
    return II_type_error
