import re
import datefinder
import numpy as np

class Utils:
    """
    Utils Class: Methods that support the search for embeddings, random strings and dates
    """
    def __init__(self, embeddings_repository):
        self.embeddings_repository = embeddings_repository
        self.num_of_embeddings = 300

    def embedding_finder(self, string):
        """
        Finds word embeddings using gensim library and Google news pretrained model for the input string. If
        the string has multiple words, it splits them an take the average embedding.
        :param string: The string that wants to find. It can be multiple words.
        :return: The embedding of the word or None if doesn't exist in the model
        """
        sum_embedding = np.zeros(self.num_of_embeddings)
        string = string.strip(' \.')
        string = re.sub(r'[^\w\s]', ' ', string)
        words_in_word = re.split('\s+', string)
        for w in words_in_word:
            embedding = self.__get_embedding_for_word(w)
            if embedding is not None:
                sum_embedding = np.add(sum_embedding, embedding)
            else:
                return None
        return sum_embedding / len(words_in_word)

    def __get_embedding_for_word(self, word):
        """
        Finds the word of the model
        :param word: The word that wants to find
        :return: The embedding of the word or None if doesn't exist in the model
        """
        embedding = self.__get_vector(word)
        if embedding is not None:
            return embedding
        else:
            return None

    def __get_vector(self, word):
        """
        Try to find the embedding
        :param word: The word that wants to find
        :return: The embedding or None
        """
        try:
            return self.embeddings_repository[word]
        except KeyError:
            return None

    @staticmethod
    def random_string_finder(string):
        """
        If a string has more than 20% of the character capitals or numbers then is a random string.
        :param string: The input string
        :return: True or False
        """
        if len(string) > 0:
            if len(re.findall(r'[A-Z0-9]', string)) / len(string) < 0.20:
                return False
            return True
        return False

    @staticmethod
    def date_finder(string):
        """
        Uses datefinder library to find if the string is data
        :param string: The input string
        :return: True or False
        """
        match = list(datefinder.find_dates(string))
        if len(match) == 0:
            return False
        return True