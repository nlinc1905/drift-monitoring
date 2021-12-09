import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chisquare


def ks_test(reference_data: pd.DataFrame, current_data: pd.DataFrame, resample: bool=True):
    """
    Calculates the p-value for the Kolmogorov-Smirnov test, which compares distributions
    of continuous variables.

    :param reference_data: expected frequencies
    :param current_data: observed frequencies
    :param resample: if True, randomly resample the reference data to make it the same size as
        the current data.  Without resampling, the p-value could be spurious, as it is highly dependent
        on sample size, and the test assumes the samples are the same size.
    """
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
            print("Datasets are same size already.")

    return ks_2samp(reference_data, current_data)[1]


def chi_square_test(reference_data: pd.DataFrame, current_data: pd.DataFrame, resample: bool=True):
    """
    Calculates the p-value for the Chi-square test, which compares observed frequencies
    to expected frequencies for categorical variables.

    :param reference_data: expected frequencies
    :param current_data: observed frequencies
    :param resample: if True, randomly resample the reference data to make it the same size as
        the current data.  Without resampling, the p-value could be spurious, as it is highly dependent
        on sample size, and the test assumes the samples are the same size.
    """
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
            print("Datasets are same size already.")

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
