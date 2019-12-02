from unittest import TestCase
import RNN_ADFA


class TestRNNAdfa(TestCase):
    def setUp(self) -> None:
        self.rnn = RNN_ADFA.RNNAdfa("120", 151)
        self.seq = ['6', '6', '63', '6', '42', '120', '6', '195', '120', '6', '6', '114', '114', '1', '1', '252', '252',
                    '252', '1', '1', '1', '1', '1', '1', '1', '1', '1', '252', '252', '252', '252', '252', '252', '252',
                    '252', '252', '252', '252', '252', '252', '252', '252', '252', '252', '252', '1', '1', '252', '1',
                    '1',
                    '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '252', '1', '1', '1', '1', '1', '1',
                    '252',
                    '252', '252', '252', '252', '252', '252', '252', '252', '252', '252', '1', '1', '1', '1', '1', '1',
                    '1',
                    '1', '1', '1', '252', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',
                    '1',
                    '1', '1', '252', '1', '252', '252', '252', '252', '252', '252', '252', '252', '252', '252', '252',
                    '252', '252', '252', '252', '252', '252', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',
                    '252',
                    '252', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']

    def tearDown(self) -> None:
        del self.rnn

    def test_get_value(self):
        temp = self.seq
        temp = temp[:30]
        s = RNN_ADFA.seq_to_str(temp)
        x = self.rnn.get_value(s)
        self.assertAlmostEqual(x, 0.00047796572)

    def test_get_value_10(self):
        temp = ["6", "6", "6", "6", "195", "6"]
        rnn = RNN_ADFA.RNNAdfa("6", 10)
        s = RNN_ADFA.seq_to_str(temp)
        x = rnn.get_value(s)
        self.assertAlmostEqual(x, 9.9283093e-01)

    def test_get_configuration(self):
        temp = self.seq
        temp = temp[:30]
        s = RNN_ADFA.seq_to_str(temp)
        x = self.rnn.get_configuration(s)
        self.assertTupleEqual(x.shape, (1, 500))
