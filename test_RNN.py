import unittest
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import Dataset
import RNN
import os
import WFA
import json
import pickle


class TestRNN(unittest.TestCase):
    """
    A network structure test
    """

    def setUp(self):
        self.hidden_output_dims = [10, 20]
        cell_type = keras.layers.LSTM
        self.max_length = 8
        self.rnn = RNN.RNN(self.hidden_output_dims, cell=cell_type)

    def test_call(self):
        x = tf.placeholder(tf.float32, shape=(None, self.max_length, 5))
        output, states = self.rnn(x)
        self.assertIsInstance(output, tf.Tensor)
        self.assertIsInstance(states, list)
        self.assertEqual(len(states), len(self.hidden_output_dims))
        for s in states:
            self.assertIsInstance(s, tf.Tensor)


class TestRNNRegression(unittest.TestCase):
    """
    A network structure test
    """

    def setUp(self):
        self.hidden_output_dims = [10, 20]
        n_alphabets = 3
        embed_dim = 10
        cell_type = keras.layers.LSTM
        self.max_length = 8
        self.rnn_regr = RNN.RNNRegression(n_alphabets,
                                          embed_dim,
                                          self.hidden_output_dims,
                                          self.max_length,
                                          rnn_cell=cell_type)

    def test_call(self):
        x = tf.placeholder(tf.float32, shape=(None, self.max_length))
        output, states = self.rnn_regr(x)
        self.assertIsInstance(output, tf.Tensor)
        self.assertIsInstance(states, list)
        self.assertEqual(len(states), len(self.hidden_output_dims))
        for s in states:
            self.assertIsInstance(s, tf.Tensor)

    def test_loss(self):
        x = tf.placeholder(tf.float32, shape=(None, self.max_length))
        y = tf.placeholder(tf.float32, shape=(None))
        loss, output = self.rnn_regr.loss(x, y)
        self.assertIsInstance(loss, tf.Tensor)
        self.assertIsInstance(output, tf.Tensor)

    def test_max_length(self):
        self.assertEqual(self.rnn_regr.max_length, self.max_length)
        with self.assertRaises(Exception):
            self.rnn_regr.max_length = 0


class TestContinuousStateMachine(unittest.TestCase):
    """
    A test of continous state machine for RNNs
    """

    def setUp(self):
        self.hidden_output_dims = [10, 20]
        alphabets = 'abc'
        embed_dim = 10
        cell_type = keras.layers.LSTM
        max_length = 8
        rnn = RNN.RNNRegression(len(alphabets),
                                embed_dim,
                                self.hidden_output_dims,
                                max_length,
                                rnn_cell=cell_type)

        self.sess = tf.Session()
        alphabet2id = Dataset.make_alphabet2id_dict(alphabets)
        self.csm = RNN.ContinuousStateMachine(rnn, self.sess, alphabet2id)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def tearDown(self):
        self.sess.close()

    def test_get_configuration(self):
        config = self.csm.get_configuration('aaa')
        self.assertIsInstance(config, np.ndarray)
        self.assertEqual(config.shape, (self.hidden_output_dims[-1],))

    def test_get_value(self):
        val = self.csm.get_value('aaa')
        self.assertIsInstance(val, np.float)


class TestContinuousStateMachine_FileLoad(unittest.TestCase):
    """
    A test of continous state machine for RNNs
    """

    def setUp(self):
        dirname = "test_rnn"
        # load wfa
        with open(os.path.join(dirname, "wfa.pickle"), "rb") as f:
            wfa: WFA.WFA = pickle.load(f)

        # load rnn setting
        with open(os.path.join(dirname, "args.json"), "r") as f:
            d = json.load(f)
            embed_dim = d["embed_dim"]
            hidden_output_dims = d["hidden_output_dims"]
            max_length = d["max_length"]

        # load
        alphabet2id = Dataset.make_alphabet2id_dict(wfa.alphabet)
        regr = RNN.RNNRegression(len(wfa.alphabet),
                                 embed_dim,
                                 hidden_output_dims,
                                 max_length)
        self.hidden_output_dims = hidden_output_dims

        tf.reset_default_graph()

        csm = RNN.ContinuousStateMachine(regr, None, alphabet2id)
        saver = tf.train.Saver(max_to_keep=0)

        self.sess = tf.Session()
        checkpoint = tf.train.latest_checkpoint(dirname)
        csm._sess = self.sess
        saver.restore(self.sess, checkpoint)
        self.csm = csm

    def tearDown(self):
        self.sess.close()

    def test_get_configuration(self):
        config = self.csm.get_configuration('aaa')
        self.assertIsInstance(config, np.ndarray)
        print(config)
        self.assertEqual(config.shape, (1, self.hidden_output_dims[-1]))

    def test_get_value(self):
        val = self.csm.get_value('aaa')
        print(val)
        self.assertIsInstance(val, np.float)


if __name__ == "__main__":
    unittest.main()
