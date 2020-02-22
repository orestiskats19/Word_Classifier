from src.main.utils.Utils import Utils
from src.main.wrapper.ScikitWrapper import ScikitWrapper
from gensim import models
import gensim.downloader as api


def classifier(word, embeddings_repository, dates_repository):
    """
    :param word: The input string from the user that will be classified
    :param embeddings_repository:
    :return: The class of the input
    """
    google_utils = Utils(embeddings_repository, 300)
    wiki_utils = Utils(dates_repository, 100)

    google_scikit_wrapper = ScikitWrapper()
    wiki_scikit_wrapper = ScikitWrapper()

    google_embedding = google_utils.embedding_finder(word)
    wiki_embedding = wiki_utils.embedding_finder(word)

    if google_embedding is not None:
        return google_scikit_wrapper.logistic_regression_predictor(google_embedding, 'google')
    elif wiki_embedding is not None:
        return wiki_scikit_wrapper.logistic_regression_predictor(wiki_embedding, 'wiki')
    elif google_utils.random_string_finder(word):
        return 'Random string'
    else:
        return 'Other'


if __name__ == '__main__':
    print("\nPlease wait the embeddings repository is loading...\n")
    embeddings_repository = models.KeyedVectors\
        .load_word2vec_format('../../../data/word_embeddings/GoogleNews-vectors-negative300.bin.gz', binary=True)
    dates_repository = api.load("glove-wiki-gigaword-100")
    while True:
        value = input("Please enter a string that you want to classify:\n")
        if value == 'no' or value == 'n':
            print('\nGoodbye')
            break
        print(f'You entered: {value}')
        print(f'The classifier predicted that the class of {value} is '
              f'{classifier(value, embeddings_repository, dates_repository)}')
        print('\nIf you would like to stop type no or n\n')
