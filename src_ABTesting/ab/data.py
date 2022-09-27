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


class BernoulliDataGenerator(object):
    def __init__(self,
                 n_groups: int = 2,
                 ctrs: list = [0.2, 0.25]):
        """
        Simple Bernoulli Generator
        Parameters:
        --------------
        n_groups: int (default=2)
            Number of ABn groups
        ctrs: python list (default=[0.2, 0.25])
            True CTR values for every AB group
        Example Usage:
        --------------
        >>> data_gen = BernoulliDataGenerator(n_groups=2, ctrs=[0.2, 0.25])
        >>> #getting 2 equal samples from 2 groups (size of every sample equals to 3)
        >>> data_gen.get_sample(update_size=3)
        """
        self.n_groups = n_groups
        self.ctrs = ctrs

    def get_sample(self,
                   update_size: int = 100):
        """
        Method to get samples from AB groups
        Parameters:
        --------------
        update_size: int (default=100)
            How many observations we want to sample from every variation
        """
        update_data = []
        for group in range(self.n_groups):
            update_data_group = np.random.binomial(1,
                                                   self.ctrs[group],
                                                   size=update_size)
            update_data.append(update_data_group)
        return update_data
