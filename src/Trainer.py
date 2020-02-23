import pandas as pd
import gensim.downloader as api
from wrapper.ScikitWrapper import ScikitWrapper
from utils.EmbeddingsUtils import EmbeddingsUtils
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from gensim import models


class Trainer:
    """
    This class trains the machine learning models
    """
    def run(self):
        """
        Trains the classifier using logistic regression and saves the models
        """
        print("Loading the emebeddings...")
        google_repository = models.KeyedVectors.load_word2vec_format(
            '../data/word_embeddings/GoogleNews-vectors-negative300.bin.gz', binary=True)
        wiki_repository = api.load("glove-wiki-gigaword-100")

        google_utils = EmbeddingsUtils(google_repository, google_repository.vector_size)
        wiki_utils = EmbeddingsUtils(wiki_repository, wiki_repository.vector_size)
        google_scikit_wrapper = ScikitWrapper(LogisticRegressionCV(class_weight='balanced',
                                                                   multi_class='multinomial',
                                                                   solver='lbfgs',
                                                                   max_iter=2000))

        wiki_scikit_wrapper = ScikitWrapper(svm.SVC(decision_function_shape='ovo'))

        print("Trains the models...")
        self.__train_helper(google_utils, google_scikit_wrapper, 'google')
        self.__train_helper(wiki_utils, wiki_scikit_wrapper, 'wiki')

    def __train_helper(self, utils, scikit_wrapper, repo_name):

        dataframe = pd.read_csv('../data/classifier_data.csv')
        embeddings = []
        classes = []

        for index, row in dataframe.iterrows():
            if repo_name == 'wiki':
                row['string'] = row['string'].lower()
            embedding = utils.embedding_finder(row['string'])
            if embedding is not None:
                embeddings.append(embedding)
                classes.append(row['class'])

        scikit_wrapper.model_trainer(embeddings, classes)
        print(f'{repo_name} model trained with accuracy: {scikit_wrapper.model_scorer(embeddings, classes)}')
        scikit_wrapper.save_ml_model(repo_name)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()
