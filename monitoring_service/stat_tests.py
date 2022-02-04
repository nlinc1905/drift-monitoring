import logging
import pandas as pd
import numpy as np
import pymc3 as pmc
from scipy.stats import ks_2samp, chisquare
from sklearn.preprocessing import MinMaxScaler


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("pymc3")
logger.propagate = False


def ks_test(reference_data: pd.Series, current_data: pd.Series, resample: bool=True):
    """
    Calculates the p-value for the Kolmogorov-Smirnov test, which compares distributions
    of continuous variables.

    :param reference_data: expected frequencies
    :param current_data: observed frequencies
    :param resample: if True, randomly resample the reference data to make it the same size as
        the current data.  Without resampling, the p-value could be spurious, as it is highly dependent
        on sample size, and the test assumes the samples are the same size.
    """
    if reference_data.dropna(axis=0).shape[0] == 0 or current_data.dropna(axis=0).shape[0] == 0:
        logging.warn("Series of length 0, returning empty string")
        return ""

    if resample:
        smaller_row_count = min(reference_data.shape[0], current_data.shape[0])
        # if reference data larger, downsample to current data size
        if smaller_row_count == current_data.shape[0]:
            reference_data = reference_data.sample(
                n=smaller_row_count,
                replace=False,
                # ignore_index=True,  # requires pandas==1.3.0
                random_state=14
            ).reset_index(drop=True)
        # if current data larger, downsample to reference data size
        elif smaller_row_count == reference_data.shape[0]:
            current_data = current_data.sample(
                n=smaller_row_count,
                replace=False,
                # ignore_index=True,  # requires pandas==1.3.0
                random_state=14
            ).reset_index(drop=True)
        else:
            logging.info("Datasets are same size already.")

    return ks_2samp(reference_data, current_data)[1]


def chi_square_test(reference_data: pd.Series, current_data: pd.Series, resample: bool=True):
    """
    Calculates the p-value for the Chi-square test, which compares observed frequencies
    to expected frequencies for categorical variables.

    :param reference_data: expected frequencies
    :param current_data: observed frequencies
    :param resample: if True, randomly resample the reference data to make it the same size as
        the current data.  Without resampling, the p-value could be spurious, as it is highly dependent
        on sample size, and the test assumes the samples are the same size.
    """
    if reference_data.dropna(axis=0).shape[0] == 0 or current_data.dropna(axis=0).shape[0] == 0:
        logging.warn("Series of length 0, returning empty string")
        return ""

    if resample:
        smaller_row_count = min(reference_data.shape[0], current_data.shape[0])
        # if reference data larger, downsample to current data size
        if smaller_row_count == current_data.shape[0]:
            reference_data = reference_data.sample(
                n=smaller_row_count,
                replace=False,
                # ignore_index=True,  # requires pandas==1.3.0
                random_state=14
            ).reset_index(drop=True)
        # if current data larger, downsample to reference data size
        elif smaller_row_count == reference_data.shape[0]:
            current_data = current_data.sample(
                n=smaller_row_count,
                replace=False,
                # ignore_index=True,  # requires pandas==1.3.0
                random_state=14
            ).reset_index(drop=True)
        else:
            logging.info("Datasets are same size already.")

    # get the observed frequencies
    ref_feature_vc = reference_data.value_counts()
    current_feature_vc = current_data.value_counts()

    # store the category labels as dict keys
    # need to create the dicts from the combined set of keys
    keys = set(list(reference_data.unique()) + list(current_data.unique()))

    ref_feature_dict = dict.fromkeys(keys, 0)
    for key, item in zip(ref_feature_vc.index, ref_feature_vc.values):
        ref_feature_dict[key] = item

    current_feature_dict = dict.fromkeys(keys, 0)
    for key, item in zip(current_feature_vc.index, current_feature_vc.values):
        current_feature_dict[key] = item

    expected_freq = [value[1] for value in sorted(ref_feature_dict.items())]
    observed_freq = [value[1] for value in sorted(current_feature_dict.items())]
    p_value = chisquare(expected_freq, observed_freq)[1]
    # TODO: find a better way to handle these
    p_value = 0.0 if (np.isnan(p_value) or np.isinf(p_value)) else p_value
    return p_value


def get_mean_and_std(dataset_a, dataset_b, min_allowed_value=1e-5):
    """
    Calculates the mean & standard deviation for datasets A and B

    :param dataset_a: numpy array of dataset A
    :param dataset_b: numpy array of dataset B
    :param min_allowed_value: (float or int) A negative or 0 value in the feature set
        returns nan or inf values for the log probability, respectively.  This causes
        MCMC to fail on a "bad initial energy" error.  Setting this parameter to a positive
        nonzero value ensures MCMC will have some kind of derivative to work with.  However,
        it also means the lowest values for a feature will be manipulated.  For BoW models,
        it should not make much difference.
    """
    dataset_a_mu = np.array(np.mean(dataset_a, axis=0))
    dataset_a_sigma = np.array(np.std(dataset_a, axis=0))
    dataset_b_mu = np.array(np.mean(dataset_b, axis=0))
    dataset_b_sigma = np.array(np.std(dataset_b, axis=0))
    # to prevent "bad initial energy" errors, ensure there are no 0.0 values that would return inf for
    #   log probability
    dataset_a_mu[dataset_a_mu <= min_allowed_value] = min_allowed_value
    dataset_a_sigma[dataset_a_sigma <= min_allowed_value] = min_allowed_value
    dataset_b_mu[dataset_b_mu <= min_allowed_value] = min_allowed_value
    dataset_b_sigma[dataset_b_sigma <= min_allowed_value] = min_allowed_value
    return dataset_a_mu, dataset_a_sigma, dataset_b_mu, dataset_b_sigma


def bayesian_a_b_test(
        a_array,
        b_array,
        min_tolerance_threshold=0.25,
        max_tolerance_threshold=0.75,
        sigma_low=1,
        sigma_high=20,
        nbr_samples_for_mcmc=2000,
        seed=14,
        cores=1,
):
    """
    Performs a Bayesian A/B test to look for differences between distribution A and distribution B.
    The test uses Pymc3's default Hamiltonian Monte Carlo (HMC) to sample from the posteriors with
    many different parameters.

    :param a_array: numpy array for dataset A
    :param b_array: numpy array for dataset B
    :param min_tolerance_threshold: (float) probability that effect size has shrunk cannot be
        > 1 - this_value (75% by default)
    :param max_tolerance_threshold: (float) probability that effect size has increased cannot be
        > this_value (75% by default)
    :param sigma_low: (float) lower bound of uniform prior standard deviation
    :param sigma_high: (float) upper bound of uniform prior standard deviation
    :param nbr_samples_for_mcmc: (int) number of samples to use for Markov Chain Monte Carlo (MCMC) sampling
    :param

    """
    a_mu, a_sigma, b_mu, b_sigma = get_mean_and_std(dataset_a=a_array, dataset_b=b_array)

    if np.all(a_array.values == a_array.values[0]) or np.all(b_array.values == b_array.values[0]):
        logging.warn("All array values are the same, cannot perform test.  Returning value of -100")
        return -100

    # start of hypothesis test
    with pmc.Model() as model:

        # prior for means of the metric to be tested
        a_mean = pmc.Normal(name='a_mean', mu=a_mu, sigma=a_sigma)
        b_mean = pmc.Normal(name='b_mean', mu=b_mu, sigma=b_sigma)

        # prior for nu (degrees of freedom in Student T's PDF) with lambda = 30 to balance
        #   nearly normal with long tailed distributions, and shifted by 1 (because lambda - 1 DoF)
        v = pmc.Exponential(name='v_minus_one', lam=1 / 29.) + 1

        # prior for standard deviations
        a_std = pmc.Uniform(name='a_std', lower=sigma_low, upper=sigma_high)
        b_std = pmc.Uniform(name='b_std', lower=sigma_low, upper=sigma_high)

        # transform prior standard deviations to precisions (precision = reciprocal of variance)
        # this will allow specifying lambda in pmc.StudentT instead of sigma, and the spread will converge
        #   towards precision as nu increases
        a_lambda = a_std**-2
        b_lambda = b_std**-2
        a = pmc.StudentT(
            name='a',
            nu=v,
            mu=a_mean,
            lam=a_lambda,
            observed=a_array
        )
        b = pmc.StudentT(
            name='b',
            nu=v,
            mu=b_mean,
            lam=b_lambda,
            observed=b_array
        )

        # calculate effect size (the diff in the means / pooled estimate of the std dev)
        diff_of_means = pmc.Deterministic(name='difference_of_means', var=b_mean - a_mean)
        diff_of_stds = pmc.Deterministic(name='difference_of_stds', var=b_std - a_std)
        effect_size = pmc.Deterministic(name='effect_size', var=diff_of_means / np.sqrt((b_std**2 + a_std**2) / 2))

        # MCMC estimation
        trace = pmc.sample(nbr_samples_for_mcmc, random_seed=seed, cores=cores, progressbar=False)

        # determine if retraining needed by scaling the effect size mean between the 94% HDI
        # this will not precisely match the probability values on the plot, but it will be close
        values = pmc.summary(
            trace[min(1000, int(0.5*nbr_samples_for_mcmc)):]
        ).loc["effect_size"][["mean", "hdi_3%", "hdi_97%"]].tolist()
        scaler = MinMaxScaler()
        effect_size_mean_scaled_btwn_94hdi = scaler.fit_transform(np.array(values).reshape(-1, 1))[0][0]
        if (
                effect_size_mean_scaled_btwn_94hdi > max_tolerance_threshold
                or effect_size_mean_scaled_btwn_94hdi < min_tolerance_threshold
        ):
            # TODO: could insert trigger here
            logging.info("Effect size has shifted, need re-training.")

        return effect_size_mean_scaled_btwn_94hdi
