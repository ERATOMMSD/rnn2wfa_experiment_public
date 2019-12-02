import unittest
import numpy as np
# import tensorflow.keras as keras
import make_prob_automaton
import WFA


class TestFunctions(unittest.TestCase):
    """
    Test functions used to make probabilistic automata
    """

    def assertNormalizedTransVec(self, vec, n_states, deg):
        self.assertEqual(len(vec), n_states)
        self.assertAlmostEqual(sum(vec), 1.)
        for x in vec:
            self.assertLessEqual(0., x)
            self.assertLessEqual(x, 1.)
        self.assertLessEqual(len([x for x in vec if x > 0.]), deg)

    def assertNormalizedTransMat(self, mat, n_states, deg):
        self.assertEqual(mat.shape, (n_states, n_states))
        for row in mat:
            self.assertNormalizedTransVec(row, n_states, deg)

    def test_make_distr(self):
        n_cells = 10
        l = make_prob_automaton.make_distr(n_cells)
        self.assertNormalizedTransVec(l, n_cells, n_cells)

    def test_make_trans_vec(self):
        n_states = 10
        deg = 3
        vec = make_prob_automaton.make_trans_vec(n_states, deg)
        self.assertNormalizedTransVec(vec, n_states, deg)

    def test_make_trans_vec_2(self):
        n_states = 10
        deg = 3
        mat = make_prob_automaton.make_trans_mat(n_states, deg)
        self.assertIsInstance(mat, np.ndarray)
        self.assertNormalizedTransMat(mat, n_states, deg)

    def test_make_prob_automaton(self):
        alphabets = "abc"
        n_states = 10
        deg = 3
        final_dist = lambda: np.random.beta(0.5, 0.5)
        wfa = make_prob_automaton.make_prob_automaton(alphabets,
                                                      n_states,
                                                      deg,
                                                      final_dist)
        self.assertIsInstance(wfa, WFA.WFA)

        self.assertIsInstance(wfa.alphabet, str)
        self.assertEqual(wfa.alphabet, alphabets)

        self.assertIsInstance(wfa.q0, np.ndarray)
        self.assertEqual(wfa.q0.shape, (1, n_states))
        self.assertNormalizedTransVec(wfa.q0.reshape(-1, ), n_states, deg)

        self.assertIsInstance(wfa.final, np.ndarray)
        self.assertEqual(wfa.final.shape, (n_states, 1))

        self.assertIsInstance(wfa.delta, dict)
        self.assertEqual(set(wfa.delta.keys()), set(alphabets))
        for mat in wfa.delta.values():
            self.assertNormalizedTransMat(mat, n_states, deg)

    def test_make_prob_automaton_2(self):
        """
        Test that there is no unbalance in generated WFAs
        :return:
        """
        alphabets = "abc"
        n_states = 10
        deg = 3
        final_dist = lambda: np.random.beta(0.5, 0.5)

        aves = []
        for i in range(50):
            wfa = make_prob_automaton.make_prob_automaton(alphabets,
                                                          n_states,
                                                          deg,
                                                          final_dist)
            aves.append(wfa.calc_average(10))
        print(aves)
        print(sum(aves) / len(aves))


if __name__ == "__main__":
    unittest.main()
