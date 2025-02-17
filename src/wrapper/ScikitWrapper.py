from joblib import dump, load


class ScikitWrapper:
    """
    Wrapper Class for the Sci-kit Library
    """
    def __init__(self, ml_model):
        """
        Initialises the Machine Learning model
        """
        self.ml_model = ml_model

    def model_trainer(self, embeddings, classes):
        """
        Trains the Machine Learning model
        :param embeddings: A list with the word embedding to classify
        :param classes: A list with the labels of the word embeddings
        """
        self.ml_model.fit(embeddings, classes)

    def model_scorer(self, embeddings, classes):
        """
        Trains the LMachine Learning model
        :param embeddings: A list with the word embedding to classify
        :param classes: A list with the labels of the word embeddings
        :return: The score of the training
        """
        return self.ml_model.score(embeddings, classes)

    def model_predictor(self, embedding, model):
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
        dump(self.ml_model, f'../data/pre_trained_models/classifier_{model}')

    def load_ml_model(self, model):
        """
        Loads the trained model
        """
        self.ml_model = load(f'../data/pre_trained_models/classifier_{model}')
