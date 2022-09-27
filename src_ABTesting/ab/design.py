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

from src_ABTesting.ab.helpers import min_sample_size_ctr, calculate_N_d


class ClassicABDesign(object):
    def __init__(self,
                 n_groups: int = 2,
                 alpha: float = 0.05,
                 beta: float = 0.8,
                 base_ctr: float = 0.3,
                 mde: float = 0.05):
        """
        Classic AB test design for CTR
        Parameters:
        --------------
        n_groups: int (default=2)
            Number of AB groups (variations)
        alpha: float (default=0.05)
            Confidence level
        beta: float (default=0.8)
            Power
        base_ctr: float (default=0.3)
            Base CTR to calc sample size (usually calculated on retro data)
        mde: float (default=0.05)
            Minimal detectable effect
        Example Usage:
        ---------------
        >>> AB = ClassicABDesign(n_groups=2,
                                 alpha=0.05,
                                 beta=0.8,
                                 base_ctr=0.3,
                                 mde=0.05)
        """
        self.n_groups = n_groups
        self.alpha = alpha
        self.beta = beta
        self.base_ctr = base_ctr
        self.mde = mde
        self.data = []
        self._calc_sample_size()
        self.is_winner_final = False
        self.is_terminate = False

    def _calc_sample_size(self):
        """
        Calculates sample size for ctr experiment
        """
        self.sample_size = min_sample_size_ctr(bcr=self.base_ctr,
                                               mde=self.mde,
                                               power=self.beta,
                                               sig_level=self.alpha)
        #print(self.sample_size)

    def update(self,
               update_data: list = []):
        """
        Updating dataset with new data
        Parameters:
        -------------
        update_data: list (default=[])
            New sample of data (shape - (n_groups; sample_size))
        Every time new data arrived checks if a current total sample size >= precalculated sample_size
        If it is true, terminates and defines if there are differences
        """
        if not len(self.data):
            self.data = np.array(update_data)
            return
        if len(self.data[0]) >= self.sample_size:
            self.is_terminate = True
        self.data = np.concatenate([self.data,
                                    np.array(update_data)], axis=1)
        self.is_winner()

    def is_winner(self):
        """
        Method to define a winner

        """
        count = self.data.sum(axis=1)
        nobs = np.array([len(self.data[0]),
                         len(self.data[0])])
        self.pval = proportions_ztest(count,
                                      nobs)[1]
        if self.pval < self.alpha:
            self.is_winner_final = True
        else:
            self.is_winner_final = False


class SequentialABDesign(object):
    def __init__(self,
                 n_groups,
                 alpha,
                 beta,
                 base_ctr,
                 mde):
        """
        https://www.evanmiller.org/sequential-ab-testing.html
        """
        self.n_groups = n_groups
        self.alpha = alpha
        self.beta = beta
        self.base_ctr = base_ctr
        self.mde = mde
        self.data = []
        self.is_winner_final = False
        self.is_terminate = False
        self._calculate_initial_params()

    def _calculate_initial_params(self):
        #self.N, self.d = calculate_N_d(alpha=self.alpha,
        #                               beta_power=self.beta,
        #                               rel_delta=self.mde,
        #                               fairness=1.0,
        #                               fast=True)
        self.N, self.d = 11141, 207

    def update(self, update_data):
        if not len(self.data):
            self.data = np.array(update_data)
            return
        self.data = np.concatenate([self.data,
                                    np.array(update_data)], axis=1)

        self.get_winner()

    def get_winner(self):
        self.C, self.T = self.data.sum(axis=1)
        if np.abs(self.T - self.C) >= 2 * np.sqrt(self.N):
            self.is_terminate = True
            self.is_winner_final = True
        elif self.T + self.C >= self.N:
            self.is_terminate = True
            self.is_winner_final = False


class BayesianABDesign(object):
    """
    Bayesian ABN test for unlimited number of groups
    Parameters
    ----------
    """

    def __init__(self, method='numeric', rule='loss',
                 alpha=0.95, alpha_prior=1, beta_prior=1,
                 resolution=500, toc=1.e-3,
                 iterations=3000, plot=False, decision_var='lift', fast=True,
                 verbose=True, sample_size_th=40000):
        self.method = method
        self.rule = rule
        self.alpha = alpha
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.resolution = resolution
        self.toc = toc
        self.iterations = iterations
        self.plot = plot
        self.decision_var = decision_var
        self.verbose = verbose
        self.fast = fast
        self.data = []
        self.is_winner_final = False
        self.is_terminate = False
        self.sample_size_th = sample_size_th

    def run(self, data):
        """
        Update priors with the observed data and run the experiment
        Parameters
        ----------
        data : `Dict(string: tuple(int, int))`
        """
        posterior = self.find_posterior(data)
        result = self.decision(posterior)
        result['p2b_winner'] = posterior['p2b_winner']
        return result

    def decision(self, posterior):
        """
        Wrapper for a decision function selector (stub).
        """
        return self.expected_loss_decision(uplift_histograms=posterior['uplift_histograms'])

    def find_posterior(self, data):
        """
        Find posterior distribution
        """
        if self.method == 'numeric':
            posterior = self.posterior_numeric(data)
        else:
            raise Exception('method not recognized')
        return posterior

    def sample_means_np(self, data):
        """
        Sample conversions directly from beta distribution.
        This method exploits conjugate distributions knowledge to speedup sampling.
        """
        mu = {}
        for group in data:
            mu[group] = np.random.beta(data[group][0] + 1, data[group][1] + 1, self.iterations)
        return mu

    def posterior_numeric(self, data):
        """
        Sample conversions and estimate cvr and uplift histograms.
        """
        n_groups = len(data)
        group_names = list(data.keys())
        bins = np.linspace(0, 1, self.resolution)
        histograms = {}
        if self.fast:
            mu = self.sample_means_np(data)
        else:
            mu = self.sample_means_mcmc(data)
        for group_idx in group_names:
            histograms[group_idx] = np.histogram(mu[group_idx], bins=bins, density=True)
        uplifts = {}
        p2b_winner = {}
        for group_idx in group_names:
            group_idx_to_compare = [i for i in group_names if i != group_idx]
            groups_to_compare = [mu[i].copy() for i in group_idx_to_compare]
            # diffs = [mu[group_idx] - mu[i] for i in group_idx_to_compare] # (maybe we'll need this for plotting)
            groups_to_compare_matrix = np.column_stack(groups_to_compare)
            uplifts[group_idx] = mu[group_idx] - groups_to_compare_matrix.max(axis=1)
            p2b_winner[group_idx] = (uplifts[group_idx] > 0).sum() / len(uplifts[group_idx])
        uplift_histograms = {}
        for group_idx in group_names:
            rvs = uplifts[group_idx]
            bins = np.linspace(np.min(rvs) - 0.2 * abs(np.min(rvs)), np.max(rvs) + 0.2 * abs(np.max(rvs)),
                               self.resolution)
            uplift_histograms[group_idx] = np.histogram(rvs, bins=bins, density=True)
        result = {'posteriors': mu, 'uplift_histograms': uplift_histograms, 'p2b_winner': p2b_winner}
        return result

    def expected_loss(self, uplift_histogram):
        """
        Calculate expected loss.
        """
        lift = uplift_histogram[1]
        frequency = uplift_histogram[0]
        lift = 0.5 * (lift[0:-1] + lift[1:])  # moving average smoothing
        auc = np.maximum(-lift, 0) * frequency
        expected_loss = np.trapz(y=auc, x=lift)
        return expected_loss

    def expected_loss_decision(self, uplift_histograms):
        """
        Calculate expected loss for each group and declare the winner if any.
        """
        expected_losses = {}
        for group_idx in uplift_histograms:
            expected_losses[group_idx] = self.expected_loss(uplift_histogram=uplift_histograms[group_idx])
        el_vector = pd.Series(expected_losses)
        current_winner = el_vector.argmin()
        winner_is_final = el_vector[current_winner] < self.toc
        return {'current_winner': current_winner, 'winner_is_final': winner_is_final,
                'expected_losses': expected_losses}

    def update(self, update_data):
        if not len(self.data):
            self.data = np.array(update_data)
            return
        self.data = np.concatenate([self.data,
                                    np.array(update_data)], axis=1)

        self.get_winner()

    def get_winner(self):
        successes = self.data.sum(axis=1)
        current_data_size = len(self.data[0])
        # print({'A': (successes[0],
        #                            current_data_size-successes[0]),
        #                      'B': (successes[1],
        #                            current_data_size-successes[1])})
        bayes_run = self.run({'A': (successes[0],
                                    current_data_size - successes[0]),
                              'B': (successes[1],
                                    current_data_size - successes[1])})
        # print(bayes_run)
        if current_data_size<1000:
            return

        if current_data_size >= self.sample_size_th:
            self.is_terminate = True
            self.is_winner_final = False
        else:
            if bayes_run['winner_is_final']:
                #self.is_terminate = True
                #self.is_winner_final = True
                # print(bayes_run)

                if bayes_run['p2b_winner']['A'] > 0.97 or bayes_run['p2b_winner']['B'] > 0.97:
                    self.is_winner_final = True
                    self.is_terminate = True
                else:
                    self.is_winner_final = False
                    self.is_terminate = True

