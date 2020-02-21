import pandas as pd
from src.main.utils.Utils import Utils
from src.main.wrapper.ScikitWrapper import ScikitWrapper


def trainer():
    """
    Trains the classifier using logistic regression and saves the models
    """
    utils = Utils()
    scikit_wrapper = ScikitWrapper()

    dataframe = pd.read_csv('../../../data/classifier_data.csv')

    embeddings = []
    classes = []

    for index, row in dataframe.iterrows():
        embedding = utils.embedding_finder(row['string'])
        if embedding is not None:
            embeddings.append(embedding)
            classes.append(row['class'])

    scikit_wrapper.logistic_regression_trainer(embeddings, classes)
    scikit_wrapper.save_ml_model()

if __name__ == '__main__':
    trainer()
