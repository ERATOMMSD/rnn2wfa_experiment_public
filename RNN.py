from typing import *
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import ContinuousStateMachine as CSM
import Dataset
import functools


class TooLongWord(Exception):
    pass


class RNN:
    def __init__(self,
                 units: List[int],
                 cell: keras.layers.RNN = keras.layers.LSTM) -> None:
        cells = [cell(n, return_sequences=True, return_state=True)
                 for n in units]
        self._cells = cells[:-1]
        self._last = cells[-1]

    def __call__(self, inputs):
        cur_inputs = inputs
        states = []
        for c in self._cells:
            cur_inputs, last_output, last_state = c(cur_inputs)
            states.append(last_state)
        output_seq, last_output, last_state = self._last(cur_inputs)
        states.append(last_state)
        return last_output, states


class RNNRegression:
    def __init__(self,
                 n_alphabets: int,
                 embed_dim: int,
                 hidden_output_dims: List[int],
                 max_length: int,
                 rnn_cell: keras.layers.RNN = keras.layers.LSTM) -> None:
        self._rnn = RNN(hidden_output_dims, cell=rnn_cell)
        self._embed = keras.layers.Embedding(n_alphabets + 1, embed_dim,
                                             mask_zero=True,
                                             input_length=max_length)
        self._dense = keras.layers.Dense(1, activation=tf.nn.relu)
        self._max_length = max_length

    def __call__(self, input_op):
        h1 = self._embed(input_op)
        h2, states = self._rnn(h1)
        o = self._dense(h2)
        return o, states

    def loss(self, input_op, label_op):
        output_op, _ = self(input_op)
        return tf.losses.mean_squared_error(label_op, output_op), output_op

    @property
    def max_length(self):
        return self._max_length


class ContinuousStateMachine(CSM.ContinuousStateMachine):

    def __init__(self,
                 rnn: RNNRegression,
                 sess: Optional[tf.Session],
                 alphabet2id: Dict[str, int]) -> None:
        self._rnn = rnn
        self._sess = sess
        self._alphabet2id = alphabet2id
        self._input_op = tf.placeholder(tf.float32,
                                        shape=(1, self._rnn.max_length))
        self.callings: Set[str] = set()
        print(self._rnn.max_length)
        # self._label_op = tf.placeholder(tf.float32, shape=(None))
        output_op, states = rnn(self._input_op)
        self._output_op = output_op
        self._state_op = states[-1]
        self.alphabet = "".join(alphabet2id.keys())

    def get_configuration(self, w: str) -> np.ndarray:
        return self._get_config_and_value(w)[0].reshape((1, -1))

    def get_value(self, w: str) -> np.float:
        return self._get_config_and_value(w)[1]

    @functools.lru_cache(maxsize=None)
    def _get_config_and_value(self, w: str) -> Tuple[np.ndarray, np.float]:
        if self._rnn.max_length is not None and len(w) > self._rnn.max_length:
            raise TooLongWord()
        self.callings.add(w)
        input_data = Dataset.word2vec(w,
                                      self._alphabet2id,
                                      self._rnn.max_length)
        input_data = input_data.reshape((1, -1))
        last_state, output = self._sess.run([self._state_op, self._output_op],
                                            feed_dict={self._input_op: input_data})

        # the shape of output is (batch size, 1)
        return last_state[0], np.float(output[0][0])

    def get_callings(self) -> int:
        return len(self.callings)
