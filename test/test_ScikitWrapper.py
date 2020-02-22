import unittest
import numpy as np
from src.main.wrapper.ScikitWrapper import ScikitWrapper


class TestScikitWrapper(unittest.TestCase):


    def test_trainer_(self):
        scikit_wrapper = ScikitWrapper()
        test_data = [np.ones(100)]
        label = ['test']
        scikit_wrapper.logistic_regression_trainer(test_data, label)
