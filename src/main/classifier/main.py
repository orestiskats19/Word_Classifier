from src.main.utils.Utils import Utils
from src.main.wrapper.ScikitWrapper import ScikitWrapper
from gensim import models

def classifier(word, embeddings_repository):
    """
    :param word: The input string from the user that will be classified
    :param embeddings_repository:
    :return: The class of the input
    """
    utils = Utils(embeddings_repository)
    scikit_wrapper = ScikitWrapper()

    word_embedding = utils.embedding_finder(word)
    if word_embedding is not None:
        return scikit_wrapper.logistic_regression_predictor(word_embedding)
    elif utils.date_finder(word):
        return 'Date'
    elif utils.random_string_finder(word):
        return 'Random string'
    else:
        return 'Other'

if __name__ == '__main__':
    print("Please wait the embeddings repository is loading...\n")
    embeddings_repository = models.KeyedVectors.load_word2vec_format('../../../data/word_embeddings/GoogleNews-vectors-negative300.bin.gz', binary=True)
    while True:
        value = input("Please enter a string that you want to classify:\n")
        if value is 'no' or 'n':
            break
        print(f'You entered: {value}')
        print(f'The classifier predicted that the class of {value} is {classifier(value, embeddings_repository)}')
        print('If you would like to stop type no or n')
