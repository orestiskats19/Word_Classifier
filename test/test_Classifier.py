import unittest
from gensim import models
import gensim.downloader as api
from src.main.classifier.main import classifier
from src.main.wrapper.ScikitWrapper import ScikitWrapper


class TestClassifier(unittest.TestCase):

    def test_that_classify_word(self):
        print("\nLoading repositories ...")
        # embeddings_repository = models.KeyedVectors \
        #     .load_word2vec_format('../../../data/word_embeddings/GoogleNews-vectors-negative300.bin.gz', binary=True)
        dates_repository = api.load("glove-wiki-gigaword-50")

        print(classifier('test', dates_repository, dates_repository))
        assert classifier('test', dates_repository, dates_repository) == 'Other'




