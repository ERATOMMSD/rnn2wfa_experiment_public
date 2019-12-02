import unittest
import QuantitativeObservationTable


class test_QuantitativeObservationTable(unittest.TestCase):
    def setUp(self) -> None:
        def f(x):
            zeros = x.count("a")
            ones = x.count("b")
            return 1.0 if zeros % 2 == 0 and ones % 2 == 0 else 0.0

        self.f = f
        self.words = ["", "ab", "aaaaaa", "ababab", "aaaabbbb"]

    def test_reconstruct_WFA_without_normalization(self):
        params = QuantitativeObservationTable.QuantitativeObservationTableParameters(False, 1e-7, 0.9, 1e-8, None)
        qot = QuantitativeObservationTable.QuantitativeObservationTable("ab", self.f, params)
        qot.add_counterexample("ab")
        qot.find_and_handle_inconsistency()
        qot.add_counterexample("aba")
        qot.find_and_handle_inconsistency()
        wfa = qot.reconstruct_WFA()
        for word in self.words:
            # print(word, self.f(word), wfa.classify_word(word))
            self.assertAlmostEqual(self.f(word), wfa.classify_word(word))

    def test_reconstruct_WFA_with_normalization(self):
        def f(x):
            zeros = x.count("a")
            ones = x.count("b")
            return 1.0 if zeros % 2 == 0 and ones % 2 == 0 else 0.0

        params = QuantitativeObservationTable.QuantitativeObservationTableParameters(True, 0.1, 0.9, 1e-5, None)
        qot = QuantitativeObservationTable.QuantitativeObservationTable("ab", self.f, params)
        qot.add_counterexample("ab")
        qot.find_and_handle_inconsistency()
        qot.add_counterexample("aba")
        qot.find_and_handle_inconsistency()
        wfa = qot.reconstruct_WFA()
        # print(wfa.show_wfa())
        for word in self.words:
            # print(word, f(word), wfa.classify_word(word))
            self.assertAlmostEqual(self.f(word), wfa.classify_word(word), 2)
