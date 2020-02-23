from src.main.utils.EmbeddingsUtils import EmbeddingsUtils
from src.main.utils.RandomStringUtils import RandomStringUtils
from src.main.wrapper.ScikitWrapper import ScikitWrapper
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from gensim import models
import gensim.downloader as api


def classifier(word, google_repository, wiki_repository):
    """
    :param word: The input string from the user that will be classified
    :param google_repository: Repository of word embeddings from google
    :param wiki_repository: Repository of word embeddings from wiki
    :return: The class of the input
    """
    google_utils = EmbeddingsUtils(google_repository, google_repository.vector_size)
    wiki_utils = EmbeddingsUtils(wiki_repository, wiki_repository.vector_size)
    random_string_utils = RandomStringUtils()

    google_scikit_wrapper = ScikitWrapper(LogisticRegressionCV(class_weight='balanced',
                                                               multi_class='multinomial',
                                                               solver='lbfgs'))
    wiki_scikit_wrapper = ScikitWrapper(svm.SVC(decision_function_shape='ovo'))

    google_embedding = google_utils.embedding_finder(word)
    wiki_embedding = wiki_utils.embedding_finder(word.lower())

    if google_embedding is not None:
        return google_scikit_wrapper.logistic_regression_predictor(google_embedding, 'google')
    elif wiki_embedding is not None:
        print("hereeeeeeeee: " + wiki_scikit_wrapper.logistic_regression_predictor(wiki_embedding, 'wiki'))
        return wiki_scikit_wrapper.logistic_regression_predictor(wiki_embedding, 'wiki')
    elif random_string_utils.random_string_finder(word):
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
