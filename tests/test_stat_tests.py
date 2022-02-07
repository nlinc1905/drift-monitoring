import unittest
import pandas as pd

from monitoring_service.stat_tests import ks_test, chi_square_test, bayesian_a_b_test


def test_ks_test():
    ref = pd.DataFrame({"a": [10.5, 16, 11.1]})
    cur = pd.DataFrame({"a": [21, 33, 22.2]})
    p = ks_test(reference_data=ref['a'], current_data=cur['a'], resample=True)
    assert round(p, 1) == 0.1


def test_chi_square_test():
    ref = pd.DataFrame({"a": [10, 16, 11]})
    cur = pd.DataFrame({"a": [20, 32, 22]})
    p = chi_square_test(reference_data=ref['a'], current_data=cur['a'], resample=True)
    assert p == 0.0


def test_bayesian_a_b_test():
    ref = pd.DataFrame({"a": [10.5, 16, 11.1]})
    cur = pd.DataFrame({"a": [21, 33, 22.2]})
    p = bayesian_a_b_test(a_array=ref['a'], b_array=cur['a'])
    assert round(p, 1) == 0.5
