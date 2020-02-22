import pandas as pd
import gensim.downloader as api
from src.main.utils.Utils import Utils
from src.main.wrapper.ScikitWrapper import ScikitWrapper
from gensim import models


def trainer():
    """
    Trains the classifier using logistic regression and saves the models
    """
    embeddings_repository = models.KeyedVectors.load_word2vec_format('../../../data/word_embeddings/GoogleNews-vectors-negative300.bin.gz', binary=True)
    dates_repository = api.load("glove-wiki-gigaword-100")

    word_utils = Utils(embeddings_repository, 300)
    date_utils = Utils(dates_repository, 100)
    scikit_wrapper = ScikitWrapper()

    dataframe = pd.read_csv('../../../data/classifier_data.csv')
    train_helper(word_utils,scikit_wrapper, dataframe, 'google')
    date_scikit_wrapper = ScikitWrapper()
    train_helper(date_utils, date_scikit_wrapper, dataframe, 'wiki')


def train_helper(utils, scikit_wrapper, dataframe, model):

    embeddings = []
    classes = []

    for index, row in dataframe.iterrows():
        embedding = utils.embedding_finder(row['string'].lower())
        if embedding is not None:
            embeddings.append(embedding)
            classes.append(row['class'])

    scikit_wrapper.logistic_regression_trainer(embeddings, classes)
    scikit_wrapper.save_ml_model(model)


if __name__ == '__main__':
    trainer()
