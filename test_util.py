import unittest
import util
import random
import itertools


class TestMethods(unittest.TestCase):
    def test_bfs_words(self):
        a = util.bfs_words("01", 3, None)
        self.assertEqual(['', '0', '1', '00', '01', '10', '11', '000', '001', '010', '011', '100', '101', '110', '111'],
                         list(a))
        b = util.bfs_words("01", None, 10)
        self.assertEqual(['', '0', '1', '00', '01', '10', '11', '000', '001', '010'], list(b))

    def test_argmax_dict(self):
        random.seed(42)
        for i in range(100):
            rn = [random.random() for _ in range(random.randint(1, 20))]
            d = {i: i for i in rn}
            m = max(rn)
            self.assertAlmostEqual(m, util.argmax_dict(d))

    def test_sample_length_from_all_words(self):
        n_alphabets = range(2, 11)
        max_lengths = [1, 2, 10, 20, 50, 100]
        for n_alphabet, max_length in itertools.product(n_alphabets, max_lengths):
            res = util.sample_length_from_all_words(n_alphabet, max_length)
            self.assertLessEqual(0, res)
            self.assertLessEqual(res, max_length)


if __name__ == "__main__":
    unittest.main()
