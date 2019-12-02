import unittest
from WFA import *
import numpy as np


class TestFunctions(unittest.TestCase):
    def test_calc_nearest_trans_mat_1(self):
        # already normalized.  The result has to be the original itself.
        x = np.array([[3.90804171e-01, 3.32969078e-01, 9.46844194e-02],
                      [6.09195829e-01, 6.67030898e-01, 2.03034454e-01],
                      [5.84671906e-10, 2.42687550e-08, 7.02281126e-01]]).T
        x1 = calc_nearest_trans_mat(x)
        np.testing.assert_array_almost_equal(x, x1, 3)

    def test_calc_nearest_trans_mat_2(self):
        # not normalized.
        x = np.array([[0.4600494, 0.23897408, -0.25668907],
                      [0.56924523, 0.40600499, -0.20251392],
                      [-0.36135676, -0.08467441, 0.04710942]]).T
        x1 = calc_nearest_trans_mat(x)
        # value condition
        np.testing.assert_array_less(0, x1)
        np.testing.assert_array_less(x1, 1)
        # sum condition
        x2 = x1.sum(axis=1)
        np.testing.assert_array_almost_equal(x2, 1)

    def test_calc_nearest_vec_of_sum1_1(self):
        # already normalized.  The result has to be the original itself.
        x = np.array([3.90804171e-01, 6.09195829e-01, 5.84671906e-10])
        x1 = calc_nearest_vec_of_sum1(x)
        np.testing.assert_array_almost_equal(x, x1, 3)

    def test_calc_nearest_vec_of_sum1_2(self):
        # not normalized
        x = np.array([3.90804171e-01, 6.09195829e-01, -5.84671906e-02])
        x1 = calc_nearest_vec_of_sum1(x)
        # value condition
        np.testing.assert_array_less(0, x1)
        np.testing.assert_array_less(x1, 1)
        # sum condition
        x2 = x1.sum()
        self.assertAlmostEqual(x2, 1)

    def test_calc_nearest_vec_of_0to1_1(self):
        # already normalized.  The result has to be the original itself.
        x = np.array([0.4, 0, 0.8])
        x1 = calc_nearest_vec_of_0to1(x)
        np.testing.assert_array_almost_equal(x, x1, 3)

    def test_calc_nearest_vec_of_0to1_2(self):
        x = np.array([0.4, 0, -0.8])
        x1 = calc_nearest_vec_of_0to1(x)
        # value condition
        np.testing.assert_array_less(0, x1)
        np.testing.assert_array_less(x1, 1)

    def test_normalize(self):
        alpha = np.array([[1.0, 3.0, 4.0]])
        beta = np.array([[2.0, 1.0, 1.0]]).T
        ma = np.array([
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 3.0],
            [1.0, 0.0, 0.0]
        ])
        mb = np.array([
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 4.0]
        ])

        wfa = WFA("ab", alpha, beta, {"a": ma, "b": mb})
        wfa = normalize(wfa)

        # check alpha
        np.testing.assert_array_less(0, wfa.q0)
        np.testing.assert_array_less(wfa.q0, 1)
        self.assertAlmostEqual(wfa.q0.sum(), 1)
        # check beta
        np.testing.assert_array_less(0, wfa.final)
        np.testing.assert_array_less(wfa.final, 1)
        # check trans
        for sigma in "ab":
            np.testing.assert_array_less(0, wfa.delta[sigma])
            np.testing.assert_array_less(wfa.delta[sigma], 1)
            np.testing.assert_array_almost_equal(wfa.delta[sigma].sum(axis=1), 1, 3)


class TestWFA(unittest.TestCase):
    """
    An example in [Balle and Mohri, 2015]
    """

    def setUp(self):
        alpha = np.array([[1, 3, 4]])
        beta = np.array([[2, 1, 1]]).T
        ma = np.array([
            [0, 0, 3],
            [0, 0, 3],
            [1, 0, 0]
        ])
        mb = np.array([
            [0, 1, 0],
            [2, 0, 0],
            [0, 0, 4]
        ])

        wfa = WFA("ab", alpha, beta, {"a": ma, "b": mb})
        self.wfa = wfa

    def test_classify_word(self):
        self.assertAlmostEqual(self.wfa.classify_word("ab"), 52)
        self.assertAlmostEqual(self.wfa.classify_word(""), 9)

    def test_calc_states(self):
        np.testing.assert_almost_equal(self.wfa.calc_states(""), np.array([[1, 3, 4]]))
        np.testing.assert_almost_equal(self.wfa.calc_states("ab"), np.array([[0, 4, 48]]))

    def test_calc_next(self):
        np.testing.assert_almost_equal(self.wfa.calc_next(np.array([[4, 0, 12]]), "b"), np.array([[0, 4, 48]]))

    def test_calc_result(self):
        np.testing.assert_almost_equal(self.wfa.calc_result(np.array([[0, 4, 48]])), 52)

    def test_calc_average(self):
        words = ["", "a", "b", "aa", "ab", "ba", "bb", "aaa", "aab", "aba", "abb", "baa", "bab", "bba", "bbb"]
        s = 0
        for word in words:
            s += self.wfa.get_value(word)
        ave = s / len(words)

        self.assertAlmostEqual(ave, self.wfa.calc_average(3))


if __name__ == "__main__":
    unittest.main()
