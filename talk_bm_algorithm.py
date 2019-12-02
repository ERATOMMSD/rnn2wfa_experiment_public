import Lstar
import equiv_query
import WFA
import QuantitativeObservationTable
import numpy as np
from typing import *
import ContinuousStateMachine
import functools


### Modify here

def e(wfa: WFA.WFA) -> equiv_query.ResultAnswerQuery.T:
    print(wfa.show_wfa())
    print("\n")
    ans = input("Is this correct?  (If yes, just type Enter.  If no, give a counterexample): ")
    if ans == "":
        return equiv_query.ResultAnswerQuery.Equivalent()
    else:
        return equiv_query.ResultAnswerQuery.Counterexample(ans)


def m(w: str) -> float:
    ans = input(f"Give the value A({w}): ")
    return float(ans)


### Modify here

class DummyEquivalenceQueryAnswerer(equiv_query.EquivalenceQueryAnswererBase):
    def answer_query(self, wfa: WFA.WFA, assert_timeout: Callable[[], None]) -> equiv_query.ResultAnswerQuery.T:
        return e(wfa)


class DummyCSM(ContinuousStateMachine.ContinuousStateMachine):
    def __init__(self, alphabet):
        self.alphabet = alphabet

    def get_configuration(self, w: str) -> np.ndarray:
        assert False

    @functools.lru_cache(None)
    def get_value(self, w: str) -> np.float:
        return m(w)

    def get_callings(self) -> int:
        return 0


params_qot = QuantitativeObservationTable.QuantitativeObservationTableParameters(normalization=False,
                                                                                 tol_rank_init=1e-5,
                                                                                 tol_rank_decay_rate=1,
                                                                                 tol_rank_lower_bound=0,
                                                                                 time_limit=None)
params = Lstar.LstarParameters(params_qot, lambda x: DummyEquivalenceQueryAnswerer(), None, None)
alphabet = "ab"

if __name__ == "__main__":
    res = Lstar.run_quantitative_lstar(DummyCSM(alphabet), params)
    print("Got ")
    print(res.wfa.show_wfa())
