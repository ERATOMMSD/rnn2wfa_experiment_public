from unittest import TestCase
import equiv_query_regr
import WFA
import numpy as np
import equiv_query
import time
import math
import random
import preserving_heapq


class TestEquivalenceQueryAnswerer_Regression(TestCase):
    def setUp(self) -> None:
        q0 = np.array([[1, 0]])
        final = np.array([[math.cos(math.radians(30)), math.sin(math.radians(30))]])
        trans = {"a": np.array([[0.5, 0.5], [0.1, 0.9]]),
                 "b": np.array([[0, 1], [1, 0]])}
        self.dummy_rnn = WFA.WFA("ab", q0, final, trans)
        self.dummy_params = equiv_query_regr.EquivalenceQueryParameters(
            comment="hoge",
            eps=0.01,
            max_length=100,
            eta=0.1,
            gamma=0.1,
            cap_m=5,
            depth_eager_search=0,
            regressor_maker=None,
            regressor_maker_name="gpr",
            time_limit=None
        )
        self.eqa = equiv_query_regr.EquivalenceQueryAnswerer(self.dummy_rnn, self.dummy_params, "hoge")

    def tearDown(self) -> None:
        del self.dummy_rnn
        del self.dummy_params
        del self.eqa

    def test__reset_timeout(self):
        pass

    def test__assert_not_timeout(self):
        self.eqa._reset_timeout()
        self.eqa.params.time_limit = 3
        self.eqa._assert_not_timeout()
        time.sleep(1)
        self.eqa._assert_not_timeout()
        time.sleep(3)
        with self.assertRaises(equiv_query.EquivalenceQueryTimedOut):
            self.eqa._assert_not_timeout()

    def test_is_around_in_wfa_config(self):
        self.dummy_params.experimental_automatic_eta = True
        random.seed(42)
        n = 1000
        c = 1e-4
        points = [(random.uniform(-c, c), random.uniform(-c, c)) for i in range(n)]
        pos = 0
        neg = 0
        beta = list(self.dummy_rnn.final.reshape((-1,)))
        print(beta)
        for p in points:
            res = self.eqa.is_around_in_wfa_config(np.array(p), np.array([0, 0]), self.dummy_rnn)
            e = self.eqa.params.e

            if (beta[0] * p[0]) ** 2 + (beta[1] * p[1]) ** 2 < (e ** 2 / 2) * 0.99:
                self.assertTrue(res, p)
                self.assertTrue(abs(beta[0] * p[0] + beta[1] * p[1]) <= e)
                pos += 1
            elif (beta[0] * p[0]) ** 2 + (beta[1] * p[1]) ** 2 > (e ** 2 / 2) * 1.01:
                self.assertFalse(res)
                neg += 1
        self.assertGreater(pos, 100)
        self.assertGreater(neg, 100)

    def test__update_p_and_get_p_delta_r(self):
        words = ["", "a", "b", "aa", "ab", "ba", "bb"]
        visited = []
        for word in words:
            visited.append(word)
            f = self.eqa._update_p_and_get_p_delta_r(visited, self.dummy_rnn)
            for v in visited:
                np.testing.assert_array_almost_equal(f(v), self.dummy_rnn.get_configuration(v))

    def test__get_criteria_string_difference(self):
        self.eqa.params.experimental_constant_allowance = True
        self.assertEqual(self.eqa._get_criteria_string_difference("x"), self.eqa.params.e)

    def test_answer_query(self):
        pass

    def test_assert_popped(self):
        self.eqa.params.experimental_bfs = False
        self.assertIsNone(self.eqa.assert_popped(0))
        with self.assertRaises(AssertionError):
            self.eqa.assert_popped(-1)
        with self.assertRaises(AssertionError):
            self.eqa.assert_popped(2)

        self.eqa.params.experimental_bfs = True
        self.eqa.params.experimental_sort_by_dist = False
        self.assertIsNone(self.eqa.assert_popped(3))
        with self.assertRaises(AssertionError):
            self.assertIsNone(self.eqa.assert_popped(-1))

        self.eqa.params.experimental_sort_by_dist = True
        self.assertIsNone(self.eqa.assert_popped(-3))
        with self.assertRaises(AssertionError):
            self.assertIsNone(self.eqa.assert_popped(1))

    def test_update_regressor(self):
        """Essentially just calling _update_p_and_get_p_delta_r"""
        pass

    def test_proceed_bfs(self):
        random.seed(42)
        self.eqa.params.experimental_automatic_eta = True
        self.eqa.params.cap_m = 2

        queue = preserving_heapq.PreservingHeapQueue()
        queue.push_with_priority("", 0)
        visited = []
        eps = 1e-5

        words = ["", "a", "b", "aa", "ab", "ba", "bb", "aaa"]

        def p_delta_r(h):
            i = words.index(h)
            random.seed(42 * i)
            return np.array([random.uniform(-eps, eps), i % 2])

        # Now (bottom, top) = (0, 0)
        popped = queue.pop()[0]
        self.assertEqual(popped, "")
        visited.append(popped)
        self.assertFalse(self.eqa.proceed_bfs(popped, p_delta_r, queue, 0, 0, visited, self.dummy_rnn)[2])
        # Now (bottom, top) = (1, 0)
        popped = queue.pop()[0]
        self.assertEqual(popped, "a")
        visited.append(popped)
        self.assertFalse(self.eqa.proceed_bfs(popped, p_delta_r, queue, 0, 0, visited, self.dummy_rnn)[2])
        # Now (bottom, top) = (1, 1)
        popped = queue.pop()[0]
        self.assertEqual(popped, "b")
        visited.append(popped)
        self.assertFalse(self.eqa.proceed_bfs(popped, p_delta_r, queue, 0, 0, visited, self.dummy_rnn)[2])
        # (bottom, top) = (2, 1)
        popped = queue.pop()[0]
        self.assertEqual(popped, "aa")
        visited.append(popped)
        self.assertFalse(self.eqa.proceed_bfs(popped, p_delta_r, queue, 0, 0, visited, self.dummy_rnn)[2])
        # (bottom, top) = (2, 2)
        popped = queue.pop()[0]
        self.assertEqual(popped, "ab")
        visited.append(popped)
        self.assertTrue(self.eqa.proceed_bfs(popped, p_delta_r, queue, 0, 0, visited, self.dummy_rnn)[2])
        # (bottom, top) = (3,2)
        popped = queue.pop()[0]
        self.assertEqual(popped, "ba")
        visited.append(popped)
        self.assertTrue(self.eqa.proceed_bfs(popped, p_delta_r, queue, 0, 0, visited, self.dummy_rnn)[2])
        # (bottom, top) = (3,3)

    def test_proceed_bfs_best_first(self):
        random.seed(42)
        self.eqa.params.experimental_automatic_eta = True
        self.eqa.params.experimental_bfs = True
        self.eqa.params.cap_m = 3

        queue = preserving_heapq.PreservingHeapQueue()
        queue.push_with_priority("", 0)
        visited = []

        w2p = {"": np.array([0, 0]),
               "a": np.array([0, 0]),
               "b": np.array([1, 0]),
               "aa": np.array([0, 0]),
               "ba": np.array([1, 0]),
               "bb": np.array([1, 0])}

        def p_delta_r(h):
            return w2p[h]

        # Now (bottom, top) = (0, 0)
        popped = queue.pop()[0]
        self.assertEqual(popped, "")
        visited.append(popped)
        self.assertFalse(self.eqa.proceed_bfs(popped, p_delta_r, queue, 0, 0, visited, self.dummy_rnn)[2])
        # Now (bottom, top) = (1, 0)
        popped = queue.pop()[0]
        self.assertEqual(popped, "a")
        visited.append(popped)
        self.assertFalse(self.eqa.proceed_bfs(popped, p_delta_r, queue, 0, 0, visited, self.dummy_rnn)[2])
        # Now (bottom, top) = (2, 0)
        popped = queue.pop()[0]
        self.assertEqual(popped, "b")
        visited.append(popped)
        self.assertFalse(self.eqa.proceed_bfs(popped, p_delta_r, queue, 0, 0, visited, self.dummy_rnn)[2])
        # (bottom, top) = (2, 1)
        popped = queue.pop()[0]
        self.assertEqual(popped, "ba")
        visited.append(popped)
        self.assertFalse(self.eqa.proceed_bfs(popped, p_delta_r, queue, 0, 0, visited, self.dummy_rnn)[2])
        # (bottom, top) = (2, 2)
        popped = queue.pop()[0]
        self.assertEqual(popped, "bb")
        visited.append(popped)
        self.assertFalse(self.eqa.proceed_bfs(popped, p_delta_r, queue, 0, 0, visited, self.dummy_rnn)[2])
        # (bottom, top) = (2, 3)

    def test_proceed_bfs_sort_by_dist(self):
        random.seed(42)
        self.eqa.params.experimental_automatic_eta = True
        self.eqa.params.experimental_bfs = True
        self.eqa.params.experimental_sort_by_dist = True
        self.eqa.params.cap_m = 3

        queue = preserving_heapq.PreservingHeapQueue()
        queue.push_with_priority("", 0)
        visited = []

        w2p = {"": np.array([0, 0]),
               "a": np.array([0, 0]),
               "b": np.array([1, 0]),
               "aa": np.array([0, 0]),
               "ba": np.array([1, 0]),
               "bb": np.array([1, 0])}

        def p_delta_r(h):
            return w2p[h]

        # Now (bottom, top) = (0, 0)
        popped = queue.pop()[0]
        self.assertEqual(popped, "")
        visited.append(popped)
        self.assertFalse(self.eqa.proceed_bfs(popped, p_delta_r, queue, 0, 0, visited, self.dummy_rnn)[2])
        # Now (bottom, top) = (1, 0)
        popped = queue.pop()[0]
        self.assertEqual(popped, "a")
        visited.append(popped)
        self.assertFalse(self.eqa.proceed_bfs(popped, p_delta_r, queue, 0, 0, visited, self.dummy_rnn)[2])
        # Now (bottom, top) = (2, 0)
        popped = queue.pop()[0]
        self.assertEqual(popped, "b")
        visited.append(popped)
        self.assertFalse(self.eqa.proceed_bfs(popped, p_delta_r, queue, 0, 0, visited, self.dummy_rnn)[2])
        # (bottom, top) = (2, 1)
        popped = queue.pop()[0]
        self.assertEqual(popped, "ba")
        visited.append(popped)
        self.assertFalse(self.eqa.proceed_bfs(popped, p_delta_r, queue, 0, 0, visited, self.dummy_rnn)[2])
        # (bottom, top) = (2, 2)
        popped = queue.pop()[0]
        self.assertEqual(popped, "bb")
        visited.append(popped)
        self.assertFalse(self.eqa.proceed_bfs(popped, p_delta_r, queue, 0, 0, visited, self.dummy_rnn)[2])
        # (bottom, top) = (2, 3)

    def test_calc_dist_min(self):
        x = {"a": np.array([1, 0, 0]), "b": np.array([0, 2, 0]), "c": np.array([0, 0, -3]), "d": np.array([0, 0, 0])}
        self.assertEqual(self.eqa.calc_dist_min("d", lambda k: x[k], 0, ["a", "b", "c"])[0], 1)

    def test_is_consistent1(self):
        """Case of a new point"""
        self.eqa.params.experimental_automatic_eta = True
        visited = ["", "a", "b"]
        x = {"": np.array([0, 0]), "a": np.array([1, 0]), "b": np.array([2, 0]), "aa": np.array([3, 0])}

        def p_delta_r(h):
            return x[h]

        res = self.eqa.is_consistent("aa", visited, p_delta_r, self.dummy_rnn, lambda _: None)
        self.assertIsInstance(res, equiv_query.ResultIsConsistent.OK)

    def test_is_consistent2(self):
        """Case of need to be refined"""
        self.eqa.params.experimental_automatic_eta = True
        self.eqa.params.depth_eager_search = -1
        visited = ["", "a", "b"]
        y = {"": np.array([0, 0]), "a": np.array([1, 0]), "b": np.array([2, 0]), "aa": np.array([1, 0])}

        def p_delta_r(h):
            return y[h]

        res = self.eqa.is_consistent("aa", visited, p_delta_r, self.dummy_rnn, lambda _: None)
        self.assertIsInstance(res, equiv_query.ResultIsConsistent.NG)

    def test_is_consistent3(self):
        """Case of no need to be refined"""
        self.eqa.params.experimental_automatic_eta = True
        self.eqa.params.depth_eager_search = -1
        visited = ["", "a", "b", "abb"]
        x = {k: self.dummy_rnn.get_configuration(k) for k in visited}

        def p_delta_r(h):
            return x[h]

        # the configuration of "a" and "abb" are equivalent because "b" just flips the configuration
        res = self.eqa.is_consistent("abb", visited, p_delta_r, self.dummy_rnn, lambda _: None)
        self.assertIsInstance(res, equiv_query.ResultIsConsistent.OK)
