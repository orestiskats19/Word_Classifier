import unittest
import gensim.downloader as api
from src.utils.EmbeddingsUtils import EmbeddingsUtils

test_repo = api.load("glove-wiki-gigaword-100")


class TestEmbeddingsUtils(unittest.TestCase):

    def test_embedding_finder_returns_embedding_when_word_is_in_corpus(self):
        utils = EmbeddingsUtils(test_repo, test_repo.vector_size)
        assert utils.embedding_finder('test') is not False

    def test_embedding_finder_returns_False_when_word_is_not_in_corpus(self):
        utils = EmbeddingsUtils(test_repo, test_repo.vector_size)
        assert utils.embedding_finder('testtesttest') is None
