from src_ABTesting.ab.data import BernoulliDataGenerator
from src_ABTesting.ab.design import ClassicABDesign, SequentialABDesign, BayesianABDesign


class SingleExperiment(object):
    def __init__(self,
                 data_generator,
                 AB,
                 update_size):
        self.data_generator = data_generator,
        self.AB = AB
        self.update_size = update_size

    def run(self):
        while True:
            update_data = self.data_generator[0].get_sample(update_size=self.update_size)
            self.AB.update(update_data=update_data)
            if self.AB.is_terminate:
                if self.AB.is_winner_final:
                    self.winner = 1
                else:
                    self.winner = 0
                self.sample_size = len(self.AB.data[0])
                break


class ExperimentSeries(object):
    def __init__(self,
                 N_experiments,
                 n_groups,
                 base_ctr,
                 true_ctrs,
                 ab_type,
                 alpha,
                 power,
                 mde,
                 update_size,
                 alpha_prior,
                 beta_prior,
                 resolution,
                 toc,
                 iterations):
        self.N_experiments = N_experiments
        self.n_groups = n_groups
        self.base_ctr = base_ctr
        self.alpha = alpha
        self.power = power
        self.mde = mde
        self.true_ctrs = true_ctrs
        self.update_size = update_size
        self.ab_type = ab_type
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.resolution = resolution
        self.toc = toc
        self.iterations = iterations

    def run_series(self):
        self.winners = []
        self.sample_sizes = []
        for experiment in range(self.N_experiments):
            # init data generator
            data_generator = BernoulliDataGenerator(n_groups=self.n_groups,
                                                    ctrs=self.true_ctrs)
            # standard AB experiment
            if self.ab_type == 'classic':
                AB = ClassicABDesign(n_groups=self.n_groups,
                                     alpha=self.alpha,
                                     beta=self.power,
                                     base_ctr=self.base_ctr,
                                     mde=self.mde)
            elif self.ab_type == 'sequential':
                AB = SequentialABDesign(n_groups=self.n_groups,
                                        alpha=self.alpha,
                                        beta=self.power,
                                        base_ctr=self.base_ctr,
                                        mde=self.mde)
            elif self.ab_type == 'bayesian':

                # init classic AB
                classicAB = ClassicABDesign(n_groups=self.n_groups,
                                            alpha=self.alpha,
                                            beta=self.power,
                                            base_ctr=self.base_ctr,
                                            mde=self.mde)
                sample_size = classicAB.sample_size

                AB = BayesianABDesign(method='numeric',
                                      rule='loss',
                                      alpha=1 - self.alpha,
                                      alpha_prior=self.alpha_prior,
                                      beta_prior=self.beta_prior,
                                      resolution=self.resolution,
                                      toc=self.toc,
                                      iterations=self.iterations,
                                      plot=False,
                                      decision_var='lift',
                                      fast=True,
                                      verbose=True,
                                      sample_size_th=sample_size)
            if self.base_ctr<0.5 and self.ab_type in ('classic', 'sequential'):
                exp = SingleExperiment(data_generator=data_generator,
                                       AB=AB,
                                       update_size=1000)
            else:
                exp = SingleExperiment(data_generator=data_generator,
                                       AB=AB,
                                       update_size=self.update_size)
            exp.run()
            self.sample_sizes.append(exp.sample_size)
            self.winners.append(exp.winner)
