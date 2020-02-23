import unittest
import numpy as np
from sklearn import svm

from src.main.wrapper.ScikitWrapper import ScikitWrapper


class TestScikitWrapper(unittest.TestCase):

    def test_that_train_an_svm_model_and_get_the_score(self):
        scikit_wrapper = ScikitWrapper(svm.SVC(decision_function_shape='ovo'))
        test_data = [np.ones(100), np.ones(100)]
        label = ['test', 'test2']
        scikit_wrapper.logistic_regression_trainer(test_data, label)
        assert scikit_wrapper.logistic_regression_scorer(test_data, label) > 0


