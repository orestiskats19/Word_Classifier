import unittest
import numpy as np
import gensim.downloader as api

from src.main.utils.EmbeddingsUtils import EmbeddingsUtils


class TestUtils(unittest.TestCase):


    def test_trainer_(self):
        dates_repository = api.load("glove-wiki-gigaword-100")
        utils = EmbeddingsUtils()
        test_data = [np.ones(100)]
        label = ['test']

