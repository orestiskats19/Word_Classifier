from sklearn.linear_model import LogisticRegressionCV
from joblib import dump, load


class ScikitWrapper:
    """
    Wrapper Class for the Sci-kit Library
    """

    def __init__(self):
        """
        Initialises the Logistic Regression model
        """
        self.ml_model = LogisticRegressionCV(class_weight='balanced', multi_class='multinomial', solver='lbfgs',
                                             max_iter=2000)

    def logistic_regression_trainer(self, embeddings, classes):
        """
        Trains the Logistic Regression model
        :param embeddings: A list with the word embedding to classify
        :param classes: A list with the labels of the word embeddings
        """
        self.ml_model.fit(embeddings, classes)
        print(self.ml_model.score(embeddings, classes))

    def logistic_regression_predictor(self, embedding, model):
        """
        Predicts the class of the word embedding
        :param embedding: The word embedding
        :return: The class of the word in a string
        """
        self.load_ml_model(model)
        return self.ml_model.predict(embedding.reshape(1, -1))[0]

    def save_ml_model(self, model):
        """
        Saves the trained model
        """
        dump(self.ml_model, f'../../../data/log_reg_classifier/log_reg_classifier_{model}')

    def load_ml_model(self, model):
        """
        Loads the trained model
        """
        self.ml_model = load(f'../../../data/log_reg_classifier/log_reg_classifier_{model}')
