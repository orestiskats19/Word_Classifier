import unittest
from src.main.utils.RandomStringUtils import RandomStringUtils


class TestRandomStringUtils(unittest.TestCase):

    def test_string_is_random_when_has_more_than_3_consecutive_vowels(self):

        string = 'aaaaaaaa'
        random_string_utils = RandomStringUtils()
        assert random_string_utils.random_string_finder(string) == True

    def test_string_is_random_when_has_more_3_consecutive_consonants(self):

        string = 'bbbbbbbb'
        random_string_utils = RandomStringUtils()
        assert random_string_utils.random_string_finder(string) == True

    def test_string_is_random_when_has_more_than_20_percent_capitals_and_numbers(self):

        string = 'bAAAAAAbb'
        random_string_utils = RandomStringUtils()
        assert random_string_utils.random_string_finder(string) == True

    def test_string_is_random_when_string_is_more_than_1_word(self):

        string = 'test aaaaa'
        random_string_utils = RandomStringUtils()
        assert random_string_utils.random_string_finder(string) == False

    def test_string_is_not_random_when_string_is_more_than_1_word(self):

        string = 'test test'
        random_string_utils = RandomStringUtils()
        assert random_string_utils.random_string_finder(string) == False


    def test_string_is_not_random_when_string_is_more_than_1_word(self):

        string = 'test test'
        random_string_utils = RandomStringUtils()
        assert random_string_utils.random_string_finder(string) == False
