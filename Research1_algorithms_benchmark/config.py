#confidence level
alpha = 0.05
#power
beta = 0.8
power = 0.8
#maximus sample size threshold for sequential AB
GLOBAL_SIZE_TH = 80000
#base conversion level
base_ctr = 0.25
#minimum detectable effect
mde = 0.05
#sample update size
update_size = 100
#number of experiment per every Base Conversion Value
N_experiments = 1000
#(TODO) number of AB groups
n_groups = 2
#relative effect (diff) to calc II type error
rel_effect = 0.05
#Bayesian
alpha_prior=1
beta_prior=1
resolution=500
toc=0.01
iterations=3000
prob_th=0.95
#define ab_type (sequential, classic or bayesian)
ab_type = 'bayesian'