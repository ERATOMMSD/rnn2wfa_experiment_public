import logging
from typing import *
import WFA
import ContinuousStateMachine
import util
from time import time
import equiv_query
import os.path
import json


mylogger = logging.getLogger("rnn2wfa").getChild("equiv_query_random")


class EquivalenceQueryParameters:
    comment: str
    max_length: int
    eps: float
    time_limit: Optional[int]
    stabilized_allowance: float
    stabilized_period: int
    shutdown_accuracy: float
    exclude_list: List[str]
    random_seed: int

    def __init__(self,
                 comment: str,
                 eps: float,
                 max_length: int,
                 train_size: int,
                 stabilized_allowance: float,
                 stabilized_period: int,
                 shutdown_accuracy: float,
                 random_seed: int,
                 time_limit: Optional[int] = None):
        self.comment = comment
        self.eps = eps
        self.max_length = max_length
        self.train_size = train_size
        self.stabilized_allowance = stabilized_allowance
        self.stabilized_period = stabilized_period
        self.shutdown_accuracy = shutdown_accuracy
        self.time_limit: Optional[int] = time_limit
        self.random_seed = random_seed


class EquivalenceQueryAnswerer(equiv_query.EquivalenceQueryAnswererBase):
    start: float
    rnn: ContinuousStateMachine.ContinuousStateMachine
    train_set: List[str]

    def __init__(self, rnn: ContinuousStateMachine.ContinuousStateMachine, params: EquivalenceQueryParameters,
                 dirname: str):
        self.rnn = rnn
        self.params = params
        self.acc_history = []
        self.dirname = dirname
        with open(os.path.join(self.dirname, "test.txt"), "r") as f:
            exclude_list = [x.strip() for x in f.readlines()]
            mylogger.info(f"exclude_list: {exclude_list}")
        self.train_set = util.make_words(self.rnn.alphabet, self.params.max_length, self.params.train_size,
                                         util.sample_length_from_all_lengths, exclude_list,
                                         self.params.random_seed)
        with open(os.path.join(self.dirname, "eqqt_train_set.json"), "w") as f:
            json.dump(self.train_set, f)

    def _reset_timeout(self):
        self.start = time()

    def _assert_not_timeout(self):
        if self.params.time_limit is not None:
            if time() - self.start > self.params.time_limit:
                raise equiv_query.EquivalenceQueryTimedOut()

    def answer_query(self, wfa: WFA.WFA, existing_ces: Iterable[str], assert_timeout: Callable[[], None]) -> Tuple[
        equiv_query.ResultAnswerQuery.T, Any]:
        self._reset_timeout()
        mylogger.info("Starting answer_query")
        self._reset_timeout()
        word2diff: Dict[str, float] = {}
        correct = 0
        for i, word in enumerate(self.train_set):
            if i % 100 == 0:
                mylogger.info(f"train{i}")
            assert_timeout()
            diff = abs(self.rnn.get_value(word) - wfa.get_value(word))
            word2diff[word] = diff
            if diff < self.params.eps:
                correct += 1
            self._assert_not_timeout()
        acc = correct / len(self.train_set)
        self.acc_history.append(acc)
        if acc > self.params.shutdown_accuracy:
            mylogger.info(f"They seems equivalent because of the sufficient accuracy: {acc}")
            return equiv_query.ResultAnswerQuery.Equivalent(), None
        if len(self.acc_history) >= self.params.stabilized_period:
            history_tail = self.acc_history[-self.params.stabilized_period:]
            if max(history_tail) - min(history_tail) < self.params.stabilized_allowance:
                mylogger.info(f"They seems Equivalent because of the stabilized accuracy: {history_tail}")
                return equiv_query.ResultAnswerQuery.Equivalent(), None
        # if they are not equivalent, returns the word of the biggest diff
        argmax = util.argmax_dict(word2diff)
        mylogger.info(
            f"Accuracy is insufficient {acc}.  The difference is {word2diff[argmax]} and the word is {argmax}.")
        return equiv_query.ResultAnswerQuery.Counterexample(argmax), None
