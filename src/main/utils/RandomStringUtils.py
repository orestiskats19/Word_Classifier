import re


class RandomStringUtils:

    def random_string_finder(self, string):
        """
        If a string has more than 20% of the character capitals or numbers then is a random string.
        :param string: The input string
        :return: True or False
        """
        if len(string) > 0:
            if len(re.findall(r'[A-Z0-9]', string)) / len(string) < 0.20:
                if self.__find_consecutive_consonants(string) or self.__find_consecutive_vowels(string):
                    return True
                else:
                    return False
            return True
        return False

    @staticmethod
    def __find_consecutive_consonants(string):
        list_of_consonants = re.findall(r'[bcdfghjklmnpqrstvwxyz]+', string, re.IGNORECASE)
        if list_of_consonants:
            if int(len(max(list_of_consonants)) >= 3):
                return True
        return False

    @staticmethod
    def __find_consecutive_vowels(string):
        list_of_vowels = re.findall(r'[aeyuio]+', string, re.IGNORECASE)
        if list_of_vowels:
            if int(len(max(list_of_vowels)) > 3):
                return True
        return False
