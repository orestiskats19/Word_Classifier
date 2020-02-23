import pandas as pd
import gensim.downloader as api
from src.main.utils.EmbeddingsUtils import EmbeddingsUtils
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from src.main.wrapper.ScikitWrapper import ScikitWrapper
from gensim import models


def trainer():
    """
    Trains the classifier using logistic regression and saves the models
    """
    print("Loading the emebeddings...")
    embeddings_repository = models.KeyedVectors.load_word2vec_format('../../../data/word_embeddings/GoogleNews-vectors-negative300.bin.gz', binary=True)
    dates_repository = api.load("glove-wiki-gigaword-100")

    google_utils = EmbeddingsUtils(embeddings_repository, 300)
    wiki_utils = EmbeddingsUtils(dates_repository, 100)
    google_scikit_wrapper = ScikitWrapper(LogisticRegressionCV(class_weight='balanced',
                                                               multi_class='multinomial', solver='lbfgs'))

    wiki_scikit_wrapper = ScikitWrapper(svm.SVC(decision_function_shape='ovo'))

    dataframe = pd.read_csv('../../../data/classifier_data.csv')

    print("Trains the models...")
    train_helper(google_utils, google_scikit_wrapper, dataframe, 'google')
    train_helper(wiki_utils, wiki_scikit_wrapper, dataframe, 'wiki')


def train_helper(utils, scikit_wrapper, dataframe, model):

    embeddings = []
    classes = []

    for index, row in dataframe.iterrows():
        if model == 'wiki':
            row['string'] = row['string'].lower()
        embedding = utils.embedding_finder(row['string'])
        if embedding is not None:
            embeddings.append(embedding)
            classes.append(row['class'])

    scikit_wrapper.logistic_regression_trainer(embeddings, classes)
    print(f'{model} model trained with accuracy: {scikit_wrapper.logistic_regression_scorer(embeddings, classes)}')
    scikit_wrapper.save_ml_model(model)


if __name__ == '__main__':
    trainer()
