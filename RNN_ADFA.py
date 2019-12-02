import ContinuousStateMachine
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
import numpy as np
from typing import *
import RNN
import functools


class KerasBatchGenerator(object):
    """Based on https://github.com/adventuresinML/adventures-in-ml-code/blob/master/keras_lstm.py"""

    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x, y


num_steps = 30
vocabulary = 152
word_to_id: Dict[str, int] = {'3': 1, '4': 2, '6': 3, '5': 4, '195': 5, '240': 6, '192': 7, '168': 8, '78': 9,
                              '197': 10, '221': 11,
                              '102': 12, '265': 13, '33': 14, '180': 15, '45': 16, '174': 17, '175': 18, '91': 19,
                              '125': 20, '140': 21,
                              '196': 22, '54': 23, '120': 24, '146': 25, '7': 26, '142': 27, '220': 28, '42': 29,
                              '63': 30, '13': 31,
                              '19': 32, '<eos>': 33, '114': 34, '11': 35, '1': 36, '85': 37, '252': 38, '201': 39,
                              '243': 40, '199': 41,
                              '309': 42, '308': 43, '122': 44, '118': 45, '219': 46, '311': 47, '141': 48, '162': 49,
                              '331': 50,
                              '10': 51, '38': 52, '191': 53, '301': 54, '27': 55, '119': 56, '258': 57, '255': 58,
                              '202': 59, '268': 60,
                              '176': 61, '200': 62, '60': 63, '158': 64, '256': 65, '41': 66, '159': 67, '160': 68,
                              '20': 69, '64': 70,
                              '12': 71, '40': 72, '242': 73, '39': 74, '307': 75, '266': 76, '66': 77, '292': 78,
                              '57': 79, '9': 80,
                              '155': 81, '143': 82, '300': 83, '340': 84, '172': 85, '148': 86, '94': 87, '157': 88,
                              '194': 89,
                              '207': 90, '97': 91, '163': 92, '314': 93, '15': 94, '183': 95, '209': 96, '213': 97,
                              '93': 98, '37': 99,
                              '133': 100, '211': 101, '229': 102, '30': 103, '117': 104, '254': 105, '208': 106,
                              '269': 107, '96': 108,
                              '226': 109, '65': 110, '203': 111, '204': 112, '212': 113, '228': 114, '231': 115,
                              '293': 116, '295': 117,
                              '320': 118, '230': 119, '144': 120, '205': 121, '214': 122, '289': 123, '83': 124,
                              '99': 125, '206': 126,
                              '272': 127, '21': 128, '26': 129, '298': 130, '104': 131, '132': 132, '179': 133,
                              '270': 134, '332': 135,
                              '128': 136, '184': 137, '234': 138, '43': 139, '77': 140, '8': 141, '110': 142,
                              '185': 143, '198': 144,
                              '224': 145, '233': 146, '259': 147, '260': 148, '264': 149, '322': 150, '75': 151, '': 0}
reversed_dictionary: Dict[int, str] = dict(zip(word_to_id.values(), word_to_id.keys()))
alphabet = "".join([chr(i) for i in range(vocabulary)])


def seq_to_str(seq: List[str]) -> str:
    return "".join([chr(word_to_id[i]) for i in seq])


def str_to_seq(s: str) -> List[str]:
    return [reversed_dictionary[ord(i)] for i in s]


def make(hidden_size=500, vocabulary=vocabulary):
    use_dropout = True
    input_op = tf.keras.Input(name="seed", shape=(num_steps,))
    embed = keras.layers.Embedding(vocabulary, hidden_size, input_length=num_steps, mask_zero=True)(input_op)
    lstm1, _, _ = keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True)(embed)
    lstm2, _, lstm2_c = keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True)(lstm1)
    if use_dropout:
        lstm2 = keras.layers.Dropout(0.5)(lstm2)
    result = keras.layers.TimeDistributed(keras.layers.Dense(vocabulary, activation='softmax'))(lstm2)
    model = keras.Model(inputs=[input_op], outputs=[result, lstm2_c])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    print(model.summary())
    return model


class RNNAdfa(ContinuousStateMachine.ContinuousStateMachine):
    model: keras.Model
    counter: int
    projection_word: str
    projection_id: int
    alphabet: str
    top_n: int
    vocabulary: int

    def __init__(self, projection_word: str, top_n: int = 151):
        self.vocabulary = top_n + 1
        model = make(500, self.vocabulary)
        model.load_weights(f"model-{top_n}.hdf5")
        self.model = model
        self.counter = 0
        self.projection_word = projection_word
        self.projection_id = word_to_id[projection_word]
        assert self.projection_id <= top_n
        self.alphabet = alphabet[:top_n + 1]
        self.top_n = top_n

    @functools.lru_cache(maxsize=None)
    def _get_config_and_value(self, w: str) -> Tuple[np.ndarray, np.float]:
        self.counter += 1
        if len(w) > num_steps:
            raise RNN.TooLongWord
        seq_id = [ord(c) for c in w]
        pad = [0] * (num_steps - len(w))
        output_rnn, s = self.model.predict(np.array([seq_id + pad]))
        am = output_rnn[0, len(w) - 1, self.projection_id]
        # print(output_rnn[0, len(w) - 1, :])
        return s, am

    def get_value(self, w: str) -> np.float:
        return self._get_config_and_value(w)[1]

    def get_callings(self) -> int:
        return self.counter

    def get_configuration(self, w: str) -> np.ndarray:
        return self._get_config_and_value(w)[0].reshape((1, -1))
