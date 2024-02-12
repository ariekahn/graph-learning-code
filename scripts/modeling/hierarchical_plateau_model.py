import pymc3 as pm
import numpy as np
import pandas as pd
import theano
import pickle

from graphs import (
    modular,
    ring_lattice,
    graph_df,
    subjects,
    subjects_lattice,
    subjects_modular
)
from roi_loader import Loader

def run():
    # project_dir = '/Users/ari/GraphLearn'
    project_dir = '/Users/ari/repos/graph-learning-data-processing'
    loader = Loader(project_dir)

    session_one = loader.load_session_one(subjects)

    is_modular = session_one['graph'] == 'Modular'
    cluster_1 = is_modular & (session_one['node'] < 5)
    cluster_2 = is_modular & (session_one['node'] > 4) & (session_one['node'] < 10)
    cluster_3 = is_modular & (session_one['node'] > 9)

    session_one['cluster'] = np.NaN
    session_one.loc[cluster_1, 'cluster'] = 1
    session_one.loc[cluster_2, 'cluster'] = 2
    session_one.loc[cluster_3, 'cluster'] = 3

    same_cluster = session_one['cluster'].shift(1) == session_one['cluster']
    same_block = session_one['run'].shift(1) == session_one['run']
    cross_cluster = is_modular & ~same_cluster & same_block
    session_one['cross_cluster'] = cross_cluster

    session_one['response_time_zscored'] = session_one.groupby(['subject', 'correct', 'run'])['response_time'].apply(lambda x: (x - x.mean())/x.std())
    session_one['valid'] = (
        (session_one['response_time_zscored'].abs() < 3)
        & session_one['correct']
        & (session_one['response_time'] < 5)
        & (session_one['response_time'] > 0.1)
    )
    session_one['session'] = 'one'

    trials = (session_one
              .loc[lambda x: x.valid]
              .loc[lambda x: x.trial_consecutive > 30]
              .reset_index(drop=True)
             )
    trials['response_time'] = trials['response_time'].astype(theano.config.floatX)
    trials['trial_consecutive'] = trials['trial_consecutive'].astype(theano.config.floatX)

    trials['subject'] = pd.Categorical(trials['subject'], np.concatenate([subjects_modular, subjects_lattice]))
    subjects_idx = trials['subject'].cat.codes.values
    n_subjects = trials['subject'].nunique()
    n_subjects_modular = len(subjects_modular)
    n_subjects_lattice = len(subjects_lattice)

    trials['movement'] = pd.Categorical(trials['movement'])
    movements_idx = trials['movement'].cat.codes.values
    n_movements = trials['movement'].nunique()

    trials['shape'] = pd.Categorical(trials['shape'])
    shapes_idx = trials['shape'].cat.codes.values
    n_shapes = trials['shape'].nunique()

    trials['is_lattice'] = trials['graph'] == 'Lattice'

    BoundNormal = pm.Bound(pm.Normal, lower=0.0)

    # with pm.Model() as hierarchical_surprisal_model:
    #     # Exponential Decay
    #     BoundNormal = pm.Bound(pm.Normal, lower=0.0)

    #     # A * e^-Bx
    #     # 300ms decay
    #     # Make sure B stays above D
    #     mu_A = BoundNormal('mu_A', mu=0.2, sigma=0.2)
    #     mu_B = pm.Bound(pm.Normal, lower=0.005)('mu_B', mu=0.01, sigma=0.01)
    #     sigma_A = pm.HalfNormal('sigma_A', 0.2)
    #     sigma_B = pm.HalfNormal('sigma_B', 0.01)
    #     A = BoundNormal('A', mu=mu_A, sigma=sigma_A, shape=n_subjects)
    #     B = BoundNormal('B', mu=mu_B, sigma=sigma_B, shape=n_subjects)
    #     fast_decay = A[subjects_idx] * pm.math.exp(-B[subjects_idx] * trials['trial_consecutive'].values)

    #     # Slow decay
    #     mu_C= BoundNormal('mu_C', mu=0.2, sigma=0.2)
    #     mu_D = pm.Bound(pm.Normal, lower=0, upper=0.005)('mu_D', mu=0.001, sigma=0.002)
    #     sigma_C = pm.HalfNormal('sigma_C', 0.2)
    #     sigma_D = pm.HalfNormal('sigma_D', 0.002)
    #     C = BoundNormal('C', mu=mu_C, sigma=sigma_C, shape=n_subjects)
    #     D = BoundNormal('D', mu=mu_D, sigma=sigma_D, shape=n_subjects)
    #     slow_decay = C[subjects_idx] * pm.math.exp(-D[subjects_idx]*trials['trial_consecutive'].values)

    #     # Based on NHB paper, these varied from about 800 to 1300
    #     # Adding dummy variables to fix one group to zero
    #     movement = pm.Normal('movement', mu=0, sigma=0.5, shape=n_movements - 1)
    #     shape = pm.Normal('shape', mu=0, sigma=0.5, shape=n_shapes - 1)
    #     movement = pm.math.concatenate([movement, [pm.math.constant(0), pm.math.constant(0)]], axis=0)
    #     shape = pm.math.concatenate([shape, [pm.math.constant(0), pm.math.constant(0)]], axis=0)

    #     graph = pm.Normal('graph', mu=0, sigma=0.2)

    #     # Intercept, hyperparameters and per-subject intercept
    #     # RTs are all going to be between 0 and 2 seconds, generally centered around 1
    #     # This seems pretty consistently around 1 as a mean
    #     # Subject sd is closer to +/-200ms
    #     mu_intercept = BoundNormal('mu_intercept', mu=1, sigma=0.3)
    #     sigma_intercept = pm.HalfNormal('sigma_intercept', 0.3)
    #     intercept = BoundNormal('intercept', mu=mu_intercept, sigma=sigma_intercept, shape=n_subjects)

    #     mu_surprisal = pm.Normal('mu_surprisal', mu=0, sigma=0.1)
    #     sigma_surprisal = pm.HalfNormal('sigma_surprisal', sigma=0.2)
    #     surprisal = pm.Normal('surprisal', mu=mu_surprisal, sigma=sigma_surprisal, shape=n_subjects_modular)
    #     for s in range(n_subjects_lattice):
    #         surprisal = pm.math.concatenate([surprisal, [pm.math.constant(0), pm.math.constant(0)]], axis=0)

    #     rt_est = (intercept[subjects_idx]
    #               + fast_decay
    #               + slow_decay
    #               + movement[movements_idx]
    #               + shape[shapes_idx]
    #               + surprisal[subjects_idx] * trials['cross_cluster'].values
    #               + graph * trials['is_lattice'].values
    #              )

    #     # Model error
    #     eps = pm.HalfCauchy('eps', 1.0)

    #     y_obs = pm.Normal('y_obs', mu=rt_est, sigma=eps, observed=trials['response_time'].values)

    #     hierarchical_surprisal_trace = pm.sample(draws=8000, tune=4000, chains=8, cores=4)
    # pm.save_trace(hierarchical_surprisal_trace, 'hierarchical_surprisal_trace')
    # with open("hierarchical_surprisal_model.pkl", "wb") as f:
    #     pickle.dump(hierarchical_surprisal_model, f)

    with pm.Model() as plateau_model:
        # Tau models the plateau timing
        # Small tau models a quick drop in RT
        # Large tau models a long delay before RT decreases
        beta_tau = pm.HalfNormal('beta_tau', sigma=1)
        tau = pm.Bound(pm.HalfCauchy, lower=0.0)('tau', beta=beta_tau, shape=n_subjects)

        # Alpha is the subject RT baseline
        mu_alpha = BoundNormal('mu_alpha', mu=1, sigma=0.3)
        sigma_alpha = pm.HalfNormal('sigma_alpha', 0.3)
        alpha = BoundNormal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=n_subjects)

        # Beta is the amount of decrease in RT due to learning
        mu_beta = BoundNormal('mu_beta', mu=0.2, sigma=0.2)
        sigma_beta = pm.HalfNormal('sigma_beta', 0.2)
        beta = BoundNormal('beta', mu=mu_beta, sigma=sigma_beta, shape=n_subjects)

        # R is the rate of RT decrease due to learning
        # Keep r bounded, reasonably small, or calculations crash due to e^-(rN)
        mu_r = pm.Bound(pm.Normal, lower=0.0, upper=0.05)('mu_r', mu=0.02, sigma=0.02)
        sigma_r = pm.HalfNormal('sigma_r', 0.02)
        r = pm.Bound(pm.Normal, lower=0.0, upper=0.05)('r', mu=mu_r, sigma=sigma_r, shape=n_subjects)

        # Potential graph effect
        graph = pm.Normal('graph', mu=0, sigma=0.2)

        # Model surprisal effect
        # Add in zeros for lattice subjects
        mu_surprisal = pm.Normal('mu_surprisal', mu=0, sigma=0.1)
        sigma_surprisal = pm.HalfNormal('sigma_surprisal', sigma=0.2)
        surprisal = pm.Normal('surprisal', mu=mu_surprisal, sigma=sigma_surprisal, shape=n_subjects_modular)
        for s in range(n_subjects_lattice):
            surprisal = pm.math.concatenate([surprisal, [pm.math.constant(0), pm.math.constant(0)]], axis=0)

        # Based on NHB paper, these varied from about 800 to 1300
        # Adding dummy variables to fix one group to zero
        movement = pm.Normal('movement', mu=0, sigma=0.5, shape=n_movements - 1)
        shape = pm.Normal('shape', mu=0, sigma=0.5, shape=n_shapes - 1)
        movement = pm.math.concatenate([movement, [pm.math.constant(0), pm.math.constant(0)]], axis=0)
        shape = pm.math.concatenate([shape, [pm.math.constant(0), pm.math.constant(0)]], axis=0)

        mu = (
            alpha[subjects_idx]
            + beta[subjects_idx] * (1 + tau[subjects_idx]) / (tau[subjects_idx] + pm.math.exp(r[subjects_idx] * trials['trial_consecutive'].values))
            + movement[movements_idx]
            + shape[shapes_idx]
            + surprisal[subjects_idx] * trials['cross_cluster'].values
            + graph * trials['is_lattice'].values
            )

        sigma = pm.HalfCauchy('eps', 1.0)

        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=trials['response_time'].values)

        plateau_trace = pm.sample(draws=8000, tune=2000, chains=8)

        pm.save_trace(plateau_trace, 'plateau_trace')
        with open("plateau_model.pkl", "wb") as f:
            pickle.dump(plateau_model, f)

if __name__=='__main__':
    run()
