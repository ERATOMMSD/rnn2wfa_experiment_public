import unittest
from typing import *

import numpy as np

import Dataset


class TestFunctions(unittest.TestCase):
    """
    A test of pure functions
    """

    def setUp(self):
        self.alphabets = 'abcd'

    def test_make_alphabet2id_dict(self):
        d = Dataset.make_alphabet2id_dict(self.alphabets)
        self.assertIsInstance(d, dict)
        self.assertEqual(len(d), len(self.alphabets))
        self.assertEqual(set(d.keys()), set(self.alphabets))
        self.assertEqual(set(d.values()), set(range(1, len(self.alphabets) + 1)))

    def test_word2vec(self):
        max_length = 10
        word = 'abc'
        d = Dataset.make_alphabet2id_dict(self.alphabets)
        vec = Dataset.word2vec(word, d, max_length)
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (max_length,))
        for i, n in enumerate(vec):
            if i < len(word):
                self.assertEqual(n, d[word[i]])
            else:
                self.assertEqual(n, 0.)

    def test_parse_alphabet_tsv(self):
        alphabet = Dataset.parse_alphabet_tsv('./test_data/alphabet.tsv')
        expected_alphabet = '01234'
        self.assertEqual(alphabet, expected_alphabet)

    def test_load_data_tsv(self):
        alphabet = '01234'
        alphabet2id = Dataset.make_alphabet2id_dict(alphabet)
        max_length = 10
        train_data: List[Tuple[np.ndarray, float]] = Dataset.load_data_tsv('./test_data/test_data.tsv', alphabet2id,
                                                                           max_length)
        self.assertEqual(len(train_data), 10)
        self.assertAlmostEqual(train_data[0][1], 0.606573)


if __name__ == "__main__":
    unittest.main()
