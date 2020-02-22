import unittest
import numpy as np
import gensim.downloader as api

from src.main.utils.Utils import Utils


class TestUtils(unittest.TestCase):


    def test_trainer_(self):
        dates_repository = api.load("glove-wiki-gigaword-100")
        utils = Utils()
        test_data = [np.ones(100)]
        label = ['test']
        scikit_wrapper.logistic_regression_trainer(test_data, label)
