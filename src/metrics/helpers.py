import numpy as np
import pandas as pd
import scipy.stats as scs
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA


def calc_stratified_std(df: pd.DataFrame = None,
                        strat_column: str = 'platform',
                        experiment_group: str = 'A',
                        metric: str = 'ARPU_inapp'
                        ):
    """
    Post Stratification (https://www.kdd.org/kdd2016/papers/files/adp0945-xieA.pdf)
    Parameters:
    --------------
    df: pd.DataFrame (default = None)
        Input dataset
    strat_column: str (default = platform)
        Stratification column for which we calc basic probabilities
    experiment_group: str (default = A)
        Name of an AB group
    metric: str (default = ARPU_inapp)
        Metric for which we need to reduce a variance
    """
    if experiment_group:
        total_mean = df[df['experiment_group']==experiment_group][metric].mean()
        platform_probs = df[strat_column].value_counts(normalize=True).to_dict()
        platform_means = df[df['experiment_group']==experiment_group].groupby(strat_column)[metric].mean().to_dict()
        platform_stds = df[df['experiment_group']==experiment_group].groupby(strat_column)[metric].std().fillna(0).to_dict()
    else:
        total_mean = df[metric].mean()
        platform_probs = df[strat_column].value_counts(normalize=True).to_dict()
        platform_means = df.groupby(strat_column)[metric].mean().to_dict()
        platform_stds = df.groupby(strat_column)[metric].std().fillna(0).to_dict()

    intra_platform_var = np.mean([(platform_stds[k]**2)*platform_probs[k] for k in platform_stds])
    inter_platform_var = np.mean([((platform_means[k]-total_mean)**2)*platform_probs[k] for k in platform_stds])
    std_stratification = np.sqrt(intra_platform_var+inter_platform_var)
    return std_stratification

def correct_alpha_boferonni(alpha, k=2):
    '''
    https://en.wikipedia.org/wiki/Multiple_comparisons_problem
    '''
    if k==2:
        return alpha
    else:
        alpha = float(alpha)/k
        return alpha

def min_sample_size(bcr, mde, power=0.8, sig_level=0.05, std=0, kind='ctr'):
    """Returns the minimum sample size to set up a split test
    Arguments:
        bcr (float): probability of success for control, sometimes
        referred to as baseline conversion rate
        mde (float): minimum change in measurement between control
        group and test group if alternative hypothesis is true, sometimes
        referred to as minimum detectable effect
        power (float): probability of rejecting the null hypothesis when the
        null hypothesis is false, typically 0.8
        sig_level (float): significance level often denoted as alpha,
        typically 0.05
    Returns:
        min_N: minimum sample size (float)
    References:
        Stanford lecture on sample sizes
        http://statweb.stanford.edu/~susan/courses/s141/hopower.pdf
    """
    if kind=='ctr':
        # standard normal distribution to determine z-values
        standard_norm = scs.norm(0, 1)

        # find Z_beta from desired power
        Z_beta = standard_norm.ppf(power)

        # find Z_alpha
        Z_alpha = standard_norm.ppf(1-sig_level/2)

        # average of probabilities from both groups
        pooled_prob = (bcr + bcr+mde) / 2

        min_N = (2 * pooled_prob * (1 - pooled_prob) * (Z_beta + Z_alpha)**2
                / mde**2)
    else:
        standard_norm = scs.norm(0, 1)

        # find Z_beta from desired power
        Z_beta = abs(standard_norm.ppf(1 - power))

        # find Z_alpha
        Z_alpha = abs(standard_norm.ppf(1 - sig_level/2))
        print(Z_beta)
        print(Z_alpha)
        min_N = (((Z_alpha+Z_beta)**2)*(std**2))/(mde**2)
        #min_N = (((Z_alpha) ** 2) * (std ** 2)) / (mde ** 2)
    return min_N

def min_sample_size_ctr(bcr, mde, power=0.8, sig_level=0.05):
    # standard normal distribution to determine z-values
    standard_norm = scs.norm(0, 1)
    # find Z_beta from desired power
    Z_beta = standard_norm.ppf(power)
    # find Z_alpha
    Z_alpha = standard_norm.ppf(1-sig_level/2)
    # average of probabilities from both groups
    pooled_prob = (bcr + bcr+mde) / 2
    min_N = (2 * pooled_prob * (1 - pooled_prob) * (Z_beta + Z_alpha)**2
             / mde**2)

    return min_N


def min_sample_size_avg(std,
                        mean_diff,
                        power=0.8,
                        sig_level=0.05):
    # standard normal distribution to determine z-values
    standard_norm = scs.norm(0, 1)
    # find Z_beta from desired power
    Z_beta = standard_norm.ppf(power)
    # find Z_alpha
    Z_alpha = standard_norm.ppf(1-sig_level/2)
    min_N = (2 * (std**2) * (Z_beta + Z_alpha)**2
             / mean_diff**2)

    return min_N

def coef_variance(x: np.array = None):
    """
    Calcs coefficient of variation on a given array
    Parameters:
    ---------------
    x: np.array (default = None)
        1-d numpy array
    Returns:
    ---------------
    coef_variance_value: np.float
        coefficient of variation
    Example usage:
    ---------------
        x = np.array([1,2,3,4,5])
        coef_variance_value = coef_variance(x=x)
    """
    try:
        coef_variance_value = np.std(x) / np.mean(x)
    except ZeroDivisionError:
        coef_variance_value = 0
    return coef_variance_value

def get_bootstrap_samples(X: np.array = None):
    """
    Returns a sample from an input array with replacement
    Parameters:
    --------------
    X: np.array (default = None)
        numpy array containing input values
    Returns:
    --------------
    sample_X: np.array
        sample from an input array with replacement
    Example usage:
    ---------------
        import numpy as np
        X = np.random.normal(10, 10, 100) #100 element size array from normal distribution
        X_sample = get_bootstrap_samples(X = X) #bootstrap sample from X
    """
    indices = np.random.randint(0, X.shape[0], size=X.shape[0])
    sample_X = X[indices]
    return sample_X

def find_nearest_plant(dat: pd.DataFrame = None,
                       catalog: pd.DataFrame = None,
                       start: str = None,
                       end: str = None,
                       pilot: list = None,
                       top: int = 3,
                       city: bool = False,
                       n_components: int = 16):
    """
    Finds 'nearest' plants to a given list of control ones
    Parameters:
    --------------
    dat: pandas DataFrame (default = None)
        Pandas dataframe containing data (metric values + calday column requ)
    Example usage:
    # catalog = pd.read_excel('xs.xls')
    # dat = pd.read_csv('data_pilot.csv', sep = ';', low_memory = False)
    # start = '20170624'
    # end = '20171222'
    # pilot = ['0020']
    # find_nearest_plant(dat, catalog, start, end, pilot)
    """

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    dat['dt'] = pd.to_datetime(dat['dt'], format='%Y-%m-%d')
    dat = dat.query("dt >= @start & dt <= @end")
    all_agg = {
        'dau': [np.sum, np.mean, np.median, np.std, coef_variance],
        'dau_ios': [np.sum, np.mean, np.median, np.std, coef_variance],
        'dau_android': [np.sum, np.mean, np.median, np.std, coef_variance],
        'dau_organic': [np.sum, np.mean, np.median, np.std, coef_variance],
        'revenue': [np.sum, np.mean, np.median, np.std, coef_variance],
        'num_purchases': [np.mean, np.median, np.std, coef_variance],
        'revenue_ios': [np.sum, np.mean, np.median, np.std, coef_variance],
        'revenue_android': [np.sum, np.mean, np.median, np.std, coef_variance],
        'num_purchases_ios': [np.sum, np.mean, np.median, np.std, coef_variance],
        'num_purchases_android': [np.sum, np.mean, np.median, np.std, coef_variance],
        'revenue_organic': [np.sum, np.mean, np.median, np.std, coef_variance],
        'num_purchases_organic': [np.sum, np.mean, np.median, np.std, coef_variance]
    }

    dat = dat.groupby('country_code').agg(all_agg).fillna(0)
    scaler_prepilot = StandardScaler()
    X = scaler_prepilot.fit_transform(dat)

    _pca = PCA(n_components=n_components)
    np.cumsum(_pca.fit(X).explained_variance_ratio_)
    X = _pca.transform(X)

    dist_matrix = pd.DataFrame(
        index=dat.index,
        columns=dat.index,
        data=squareform(pdist(X)))

    store_dict = {}
    pilot = [x for x in pilot]

    for store in pilot:
        try:
            temp = dist_matrix.loc[~dist_matrix.index.isin(pilot)]
            imp = temp.loc[:, store].sort_values(
                ascending=True)[:10].index.values

            if city:
                city = catalog['City'][catalog['Store №'].values == store].values[0]
                store_control = catalog['Store №'][catalog['City'] == city].values
                store_control = list(np.intersect1d(store_control, imp))[:top]
            else:
                store_control = imp[:top]
            store_dict[store] = store_control

        except Exception as e:
            raise e

    return store_dict
