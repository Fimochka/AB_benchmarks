from math import ceil
from math import exp as math_exp
from math import isnan, log, pi, sqrt
import scipy.stats as scs
import scipy.special
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
import pandas as pd
from scipy.stats import beta
import matplotlib.pyplot as plt
import warnings

MAX_BARRIER = 5000
MAX_CONVERSIONS = 800000


def min_sample_size_ctr(bcr, mde, power=0.8, sig_level=0.05):

    """ Based on https://www.evanmiller.org/ab-testing/sample-size.html

    Args:
        alpha (float): How often are you willing to accept a Type I error (false positive)?
        power (float): How often do you want to correctly detect a true positive (1-beta)?
        bcr (float): Base conversion rate
        mde (float): Minimum detectable effect, relative to base conversion rate.

    """
    # standard normal distribution to determine z-values
    standard_norm = scs.norm(0, 1)
    delta = bcr*mde
    # find Z_alpha
    t_alpha2 = standard_norm.ppf(1.0-sig_level/2)
    # find Z_beta from desired power
    t_beta = standard_norm.ppf(power)
    sd1 = np.sqrt(2 * bcr * (1.0 - bcr))
    sd2 = np.sqrt(bcr * (1.0 - bcr) + (bcr + delta) * (1.0 - bcr - delta))

    return int((t_alpha2 * sd1 + t_beta * sd2) * (t_alpha2 * sd1 + t_beta * sd2) / (delta * delta))



def min_sample_size_avg(std,
                        mean_diff,
                        power=0.8,
                        sig_level=0.05):
    # standard normal distribution to determine z-values
    standard_norm = scs.norm(0, 1)
    # find Z_beta from desired power
    Z_beta = standard_norm.ppf(power)
    # find Z_alpha
    Z_alpha = standard_norm.ppf(1 - sig_level / 2)
    min_N = (2 * (std ** 2) * (Z_beta + Z_alpha) ** 2
             / mean_diff ** 2)

    return min_N


def calculate_N(alpha, beta_power, rel_mde, fairness=1.0, fast=True):
    N, d = calculate_N_d(alpha, beta_power, rel_mde, fairness=fairness, fast=fast)
    return N


def calculate_d(alpha, N, fairness=1.0, fast=True):
    d = min(
        _d_for_n(1, MAX_BARRIER - 1, alpha, N, fairness=fairness, fast=fast),
        _d_for_n(2, MAX_BARRIER, alpha, N, fairness=fairness, fast=fast),
    )
    return d


def _get_p(rel_delta, fairness):
    return 1.0 / (1.0 + (1 + rel_delta) * fairness)


def get_significance(d, N, fairness=1.0, fast=True):
    return _total_ruin_prob(d, N, _get_p(0.0, fairness), fast=fast)


def get_power(d, N, rel_mde, fairness=1.0, fast=True):
    return _total_ruin_prob(d, N, _get_p(rel_mde, fairness), fast=fast)


def get_fairness(control_visitors, alt_visitors):
    assert control_visitors > 0 and alt_visitors > 0, "Some traffic should be presented"
    # fairness is not probability, it's probability ratio
    # suppose: prob(control) = visitors(control) / (visitors(alt) + visitors(countrol))
    # and: prob(alt) = visitors(alt) / (visitors(alt) + visitors(countrol))
    # then: prob(alt) / prob(control) = visitors(alt) / visitors(control)
    fairness = alt_visitors / float(control_visitors)
    return fairness


def _safe_middle(left, right):
    # 948, 1000 -> 974, but 949, 1000 -> 973 and 950, 1000 -> 974
    return left + 2 * ((right - left) // 4)


def calculate_N_d(alpha, beta_power, rel_delta, fairness=1.0, fast=True):
    null_p = _get_p(0.0, fairness)
    alt_p = _get_p(rel_delta, fairness)

    best_odd_d, odd_n = _binary_search(
        1, MAX_BARRIER - 1, alpha, beta_power, null_p, alt_p, fast=fast
    )
    best_even_d, even_n = _binary_search(
        2, MAX_BARRIER, alpha, beta_power, null_p, alt_p, fast=fast
    )

    if odd_n is None or even_n < odd_n:
        return even_n, best_even_d
    else:
        return odd_n, best_odd_d


def _ruin_params(d, n, fast=True):
    k = (n + d) // 2
    if fast and n != d:  # approximation doesn't work when d == n, because k / n should be < 1
        # use Stirling's approximation: https://en.wikipedia.org/wiki/Stirling%27s_approximation
        prefix = float(d) / n / sqrt(2 * pi * n)
        a = k / float(n)
        log_binom = -((k + 0.5) * log(a) + (n - k + 0.5) * log(1 - a))
        return k, prefix, log_binom
    else:
        # precise calculation
        prefix = float(d) / (n * (n + 1))
        log_binom = -scipy.special.betaln(k + 1, n - k + 1)
        return k, prefix, log_binom


def _ruin_prob(d, k, log_binom, log_p, log_q):
    return math_exp(log_binom + (k - d) * log_p + k * log_q)


def _total_ruin_prob(d, N, p, fast=True):
    if d <= 0:
        # gambler has no coins and it's guaranteed to loose
        return 1.0

    log_p = log(p)
    log_q = log(1.0 - p)

    cdf = 0.0
    for n in range(d, N + 1, 2):
        k, prefix, log_binom = _ruin_params(d, n, fast=fast)
        cdf += prefix * _ruin_prob(d, k, log_binom, log_p, log_q)

    return cdf


def _binary_search(d_lo, d_hi, alpha, power_level, null_p, alt_p, fast=True):
    log_null_p = log(null_p)
    log_null_q = log(1.0 - null_p)

    log_alt_p = log(alt_p)
    log_alt_q = log(1.0 - alt_p)

    d = _safe_middle(d_lo, d_hi)
    found_n = MAX_CONVERSIONS
    while d_lo < d_hi:
        null_cdf = 0.0
        alt_cdf = 0.0

        for n in range(d, MAX_CONVERSIONS + 1, 2):
            k, prefix, log_binom = _ruin_params(d, n, fast=fast)
            null_cdf += prefix * _ruin_prob(d, k, log_binom, log_null_p, log_null_q)
            alt_cdf += prefix * _ruin_prob(d, k, log_binom, log_alt_p, log_alt_q)

            if isnan(null_cdf) or isnan(alt_cdf):
                break

            if alt_cdf > power_level:
                if null_cdf < alpha:
                    d_hi = d
                    found_n = n
                else:
                    d_lo = d + 2
                break
            elif null_cdf > alpha:
                d_lo = d + 2
                break
        else:
            return d, MAX_CONVERSIONS

        if isnan(null_cdf) or isnan(alt_cdf):
            return d, MAX_CONVERSIONS

        d = _safe_middle(d_lo, d_hi)

    return d, found_n


def _d_for_n(d_lo, d_hi, alpha, N, fairness=1.0, fast=True):
    p = _get_p(0.0, fairness)

    while d_lo < d_hi:
        d = _safe_middle(d_lo, d_hi)
        cdf = _total_ruin_prob(d, N, p, fast=fast)

        if cdf > alpha:
            d_lo = d + 2
        else:
            d_hi = d

    return d_lo


def beta_to_bin(positive, negative):
    """
    Convert (alpha, beta) parameter set of a beta distribution to a binary integer vector
    """
    return np.concatenate([np.ones(positive), np.zeros(negative)])
