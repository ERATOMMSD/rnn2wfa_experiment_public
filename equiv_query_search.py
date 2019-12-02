import logging
from typing import *
import WFA
import ContinuousStateMachine
from time import time
import equiv_query

mylogger = logging.getLogger("rnn2wfa").getChild("equiv_query_search")


class EquivalenceQueryParameters:
    comment: str
    time_limit: Optional[int]
    quit_number: int
    e: float
    experimental_reset: bool
    experimental_disable_equivalent: bool
    experimental_skip_existing: bool
    experimental_halt_long_word: Optional[int]

    def __init__(self,
                 comment: str = "",
                 e: float = 0.05,
                 time_limit: Optional[int] = None,
                 quit_number: int = 1000,
                 experimental_reset=False,
                 experimental_disable_equivalent=False,
                 experimental_skip_existing=False,
                 experimental_halt_long_word=None):
        self.comment = comment
        self.e = e
        self.time_limit: Optional[int] = time_limit
        self.quit_number: int = quit_number
        self.experimental_reset = experimental_reset
        self.experimental_disable_equivalent = experimental_disable_equivalent
        self.experimental_skip_existing = experimental_skip_existing
        self.experimental_halt_long_word = experimental_halt_long_word


class EquivalenceQueryAnswerer(equiv_query.EquivalenceQueryAnswererBase):
    start: float
    rnn: ContinuousStateMachine.ContinuousStateMachine
    queue: List[str]

    def __init__(self, rnn: ContinuousStateMachine.ContinuousStateMachine, params: EquivalenceQueryParameters,
                 dirname: str):
        self.rnn = rnn
        self.params = params
        self.queue = [""]
        self.cnt = 0

    def _reset_timeout(self):
        self.start = time()

    def _assert_not_timeout(self):
        if self.params.time_limit is not None:
            if time() - self.start > self.params.time_limit:
                raise equiv_query.EquivalenceQueryTimedOut()

    def answer_query(self, wfa: WFA.WFA, existing_ces: Iterable[str], assert_timeout: Callable[[], None]) -> Tuple[
        equiv_query.ResultAnswerQuery.T, Any]:
        mylogger.info("Starting answer_query")
        self._reset_timeout()
        previous_queue_num = 0
        if self.params.experimental_reset:
            previous_queue_num = self.cnt
            self.cnt = 0
            self.queue = [""]
        self.cnt = 0
        while self.queue:
            assert_timeout()
            h = self.queue.pop(0)
            if self.params.experimental_halt_long_word is not None \
                    and self.params.experimental_halt_long_word < len(h):
                raise equiv_query.TooLongWordExceptionAndHalted(h)
            if abs(self.rnn.get_value(h) - wfa.classify_word(h)) >= self.params.e:
                mylogger.debug(f"Found a counterexample '{h}' in usual search")
                if self.params.experimental_skip_existing and h in existing_ces:
                    mylogger.debug(f"Skipping counterexample '{h}'")
                else:
                    # put back h
                    self.queue.insert(0, h)
                    return equiv_query.ResultAnswerQuery.Counterexample(h), None
            self.queue += [h + sigma for sigma in self.rnn.alphabet]
            self.cnt += 1
            threshold = self.params.quit_number
            threshold += previous_queue_num if self.params.experimental_reset else 0
            if self.cnt > threshold and (not self.params.experimental_disable_equivalent):
                break

        mylogger.info("Seems equivalent")
        return equiv_query.ResultAnswerQuery.Equivalent(), None
