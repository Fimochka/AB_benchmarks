import numpy as np
import pandas as pd

from src_ABTesting.ab.design import ClassicABDesign, SequentialABDesign, BayesianABDesign
from src_ABTesting.ab.experiment import SingleExperiment, ExperimentSeries
from src_ABTesting.ab.data import BernoulliDataGenerator

from config import alpha, beta, power, GLOBAL_SIZE_TH, base_ctr, \
                   mde, update_size, N_experiments, n_groups, \
                   alpha_prior, beta_prior, resolution, toc, iterations, prob_th, rel_effect, ab_type


params = 'rel_effect_{}_toc_{}_a_{}_b_{}'.format(rel_effect, toc,
                                   alpha_prior,
                                   beta_prior)

if __name__=='__main__':
    total_stats_df = pd.DataFrame()
    final_df = pd.DataFrame()

    ctrs = []
    itype_errors = []
    size_avg = []
    size_total = []
    size_std = []

    for base_ctr in np.arange(0.05, 0.95, 0.01):
        print(base_ctr)
        base_ctr_treatment = base_ctr+base_ctr*rel_effect
        if base_ctr_treatment>1:
            base_ctr_treatment=1
        exp_series = ExperimentSeries(N_experiments=N_experiments,
                                      n_groups=n_groups,
                                      base_ctr=base_ctr,
                                      true_ctrs=[base_ctr,
                                                 base_ctr_treatment],
                                      ab_type=ab_type,
                                      alpha=alpha,
                                      power=power,
                                      mde=mde,
                                      update_size=update_size,
                                      alpha_prior=alpha_prior,
                                      beta_prior=beta_prior,
                                      resolution=resolution,
                                      toc=toc,
                                      iterations=iterations)
        exp_series.run_series()
        I_type_error = 1-np.sum(exp_series.winners) / float(len(exp_series.winners))
        sample_mean = np.mean(exp_series.sample_sizes)
        print(np.sum(exp_series.sample_sizes))
        sample_sum = np.sum(exp_series.sample_sizes)
        sample_std = np.std(exp_series.sample_sizes)
        ctrs.append(base_ctr)
        itype_errors.append(I_type_error)
        size_avg.append(sample_mean)
        size_total.append(sample_sum)
        size_std.append(sample_std)
        print("II type error: {}".format(I_type_error))

        current_ctr_stats_df = pd.DataFrame({'sample_size': exp_series.sample_sizes})
        current_ctr_stats_df['ctr'] = base_ctr
        current_ctr_stats_df['ab_type'] = ab_type
        final_df = pd.concat([final_df,
                              current_ctr_stats_df])

    total_stats_ = pd.DataFrame({'base_ctr': ctrs,
                                 'ii_type_error': itype_errors,
                                 'sample_size_total': size_total,
                                 'sample_size_avg': size_avg,
                                 'sample_size_std': size_std
                                 })
    #total_stats_['sample_size_avg'] = total_stats_['sample_size_total'].apply(lambda x: int(float(x) / N_experiments), 1)
    #total_stats_['sample_size_std'] = 0
    total_stats_['mde'] = mde
    total_stats_['alpha'] = alpha
    total_stats_['beta'] = beta
    total_stats_['base_ctr_true_relative_diff'] = rel_effect

    total_stats_df = pd.concat([total_stats_, total_stats_df])
    total_stats_df[['mde',
                    'alpha',
                    'beta',
                    'base_ctr',
                    'ii_type_error',
                    'sample_size_total',
                    'sample_size_avg',
                    'sample_size_std',
                    'base_ctr_true_relative_diff']].round(4).to_csv("reports/II_TYPE_ERROR_{ab_type}_{params}.csv".format(ab_type=ab_type,
                                                                                                                          params=params),
                                                                    index=None)

