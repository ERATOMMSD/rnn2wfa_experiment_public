from unittest import TestCase
import numpy as np
import WFA
import equiv_query_search
import time
import equiv_query


class TestEquivalenceQueryAnswerer(TestCase):
    def setUp(self) -> None:
        # Always returns 0
        q0 = np.array([[1, 0]])
        final = np.array([[0, 0]]).T
        trans = {"a": np.eye(2),
                 "b": np.eye(2)}
        self.dummy_wfa = WFA.WFA("ab", q0, final, trans)

        # Returns 1 only when "ab"
        q0 = np.array([[1, 0, 0]])
        final = np.array([[0, 0, 1]]).T
        trans = {"a": np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
                 "b": np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])}
        self.dummy_rnn = WFA.WFA("ab", q0, final, trans)

        # Returns 1 only when "ab" or "aa"
        q0 = np.array([[1, 0, 0]])
        final = np.array([[0, 0, 1]]).T
        trans = {"a": np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
                 "b": np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])}
        self.dummy_rnn2 = WFA.WFA("ab", q0, final, trans)

        self.dummy_params = equiv_query_search.EquivalenceQueryParameters(
            comment="",
            quit_number=5
        )
        self.eqa = equiv_query_search.EquivalenceQueryAnswerer(self.dummy_rnn, self.dummy_params, "hoge")

    def tearDown(self) -> None:
        del self.dummy_rnn
        del self.dummy_params
        del self.eqa

    def test__assert_not_timeout(self):
        self.eqa._reset_timeout()
        self.eqa.params.time_limit = 3
        self.eqa._assert_not_timeout()
        time.sleep(1)
        self.eqa._assert_not_timeout()
        time.sleep(3)
        with self.assertRaises(equiv_query.EquivalenceQueryTimedOut):
            self.eqa._assert_not_timeout()

    def test_answer_query_no_reset(self):
        res = self.eqa.answer_query(self.dummy_wfa, lambda: None)
        self.assertIsInstance(res[0], equiv_query.ResultAnswerQuery.Counterexample)
        self.assertEqual(res[0].content, "ab")
        res = self.eqa.answer_query(self.dummy_wfa, lambda: None)
        self.assertIsInstance(res[0], equiv_query.ResultAnswerQuery.Counterexample)
        self.assertEqual(res[0].content, "ab")
        res = self.eqa.answer_query(self.dummy_rnn, lambda: None)
        self.assertIsInstance(res[0], equiv_query.ResultAnswerQuery.Equivalent)

    def test_answer_query_no_reset2(self):
        res = self.eqa.answer_query(self.dummy_wfa, lambda: None)
        self.assertIsInstance(res[0], equiv_query.ResultAnswerQuery.Counterexample)
        self.assertEqual(res[0].content, "ab")
        res = self.eqa.answer_query(self.dummy_wfa, lambda: None)
        self.assertIsInstance(res[0], equiv_query.ResultAnswerQuery.Counterexample)
        self.assertEqual(res[0].content, "ab")
        res = self.eqa.answer_query(self.dummy_rnn2, lambda: None)
        self.assertIsInstance(res[0], equiv_query.ResultAnswerQuery.Equivalent)

    def test_answer_query_reset(self):
        self.eqa.params.experimental_reset = True
        res = self.eqa.answer_query(self.dummy_wfa, lambda: None)
        self.assertIsInstance(res[0], equiv_query.ResultAnswerQuery.Counterexample)
        self.assertEqual(res[0].content, "ab")
        res = self.eqa.answer_query(self.dummy_wfa, lambda: None)
        self.assertIsInstance(res[0], equiv_query.ResultAnswerQuery.Counterexample)
        self.assertEqual(res[0].content, "ab")
        res = self.eqa.answer_query(self.dummy_rnn, lambda: None)
        self.assertIsInstance(res[0], equiv_query.ResultAnswerQuery.Equivalent)

    def test_answer_query_reset2(self):
        self.eqa.params.experimental_reset = True
        res = self.eqa.answer_query(self.dummy_wfa, lambda: None)
        self.assertIsInstance(res[0], equiv_query.ResultAnswerQuery.Counterexample)
        self.assertEqual(res[0].content, "ab")
        res = self.eqa.answer_query(self.dummy_wfa, lambda: None)
        self.assertIsInstance(res[0], equiv_query.ResultAnswerQuery.Counterexample)
        self.assertEqual(res[0].content, "ab")
        res = self.eqa.answer_query(self.dummy_rnn2, lambda: None)
        self.assertIsInstance(res[0], equiv_query.ResultAnswerQuery.Counterexample)
        self.assertEqual(res[0].content, "aa")
