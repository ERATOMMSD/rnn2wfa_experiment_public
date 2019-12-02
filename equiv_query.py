from typing import *
import abc
import ContinuousStateMachine
import WFA


class EquivalenceQueryTimedOut(Exception):
    pass


class TooLongWordExceptionAndHalted(Exception):
    def __init__(self, word):
        self.word = word


class ResultIsConsistent:
    class OK(NamedTuple):
        pass

    class NG(NamedTuple):
        pass

    class Counterexample(NamedTuple):
        content: str

    T = Union[OK, NG, Counterexample]


class ResultAnswerQuery:
    class Equivalent(NamedTuple):
        pass

    class Counterexample(NamedTuple):
        content: str

    T = Union[Equivalent, Counterexample]


class EquivalenceQueryAnswererBase(abc.ABC):
    @abc.abstractmethod
    def answer_query(self, wfa: WFA.WFA, existing_ces: Iterable[str], assert_timeout: Callable[[], None]) -> Tuple[
        ResultAnswerQuery.T, Any]:
        pass
