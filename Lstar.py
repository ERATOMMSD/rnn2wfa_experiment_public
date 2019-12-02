import QuantitativeObservationTable
import ContinuousStateMachine
from time import time
import equiv_query
import logging
import WFA
from typing import *
import dataclasses
import pickle
import os
import os.path
import RNN

mylogger = logging.getLogger("rnn2wfa").getChild("Lstar")
mylogger.info("loaded")


class LstarTimedOut(Exception):
    pass


TEquivalenceQueryMaker = Callable[[ContinuousStateMachine.ContinuousStateMachine],
                                  equiv_query.EquivalenceQueryAnswererBase]


class LstarParameters:
    time_limit: Optional[int]
    params_qot: QuantitativeObservationTable.QuantitativeObservationTableParameters
    eqqa_maker: TEquivalenceQueryMaker
    save_process: Optional[str]

    def __init__(self, params_qot: QuantitativeObservationTable.QuantitativeObservationTableParameters,
                 eqqa_maker: TEquivalenceQueryMaker,
                 time_limit: Optional[int],
                 save_process: Optional[str]
                 ):
        self.params_qot = params_qot
        self.eqqa_maker = eqqa_maker
        self.time_limit = time_limit
        self.save_process = save_process


class Statistics:
    extracting_time: float
    size_wfa: int
    periods_lstar: List[float]
    calling_mem: List[int]
    eqq_time: float
    table_time: float
    add_ce_time: float
    stats_in_eqq: List[Any]

    def __init__(self):
        self.extracting_time = 0
        self.size_wfa = 0
        self.periods_lstar = []
        self.calling_mem = []
        self.eqq_time = 0
        self.table_time = 0
        self.add_ce_time = 0
        self.stats_in_eqq = []

    def to_dict(self):
        return {"extracting_time": self.extracting_time,
                "size_wfa": self.size_wfa,
                "periods_lstar": self.periods_lstar,
                "calling_mem": self.calling_mem,
                "eqq_time": self.eqq_time,
                "table_time": self.table_time,
                "add_ce_time": self.add_ce_time,
                "stats_in_eqq": self.stats_in_eqq}


@dataclasses.dataclass
class LStarResult:
    wfa: WFA.WFA
    stat: Statistics


def run_quantitative_lstar(target: ContinuousStateMachine.ContinuousStateMachine,
                           params: LstarParameters) -> LStarResult:
    mylogger.info("running")
    stat = Statistics()
    params_qot = params.params_qot
    eqqa_maker = params.eqqa_maker
    eqq = eqqa_maker(target)
    table = QuantitativeObservationTable.QuantitativeObservationTable(target.alphabet, target.get_value, params_qot)
    wfa = None
    lstar_start = time()
    eqq_time = 0
    table_time = 0
    add_ce_time = 0
    stats_in_eqq = []

    def add_stat():
        stat.calling_mem.append(target.get_callings())
        stat.periods_lstar.append(time() - lstar_start)
        if params.save_process is not None:
            with open(os.path.join(params.save_process, f"wfa{len(stat.calling_mem) - 1}.pickle"), "wb") as f:
                pickle.dump(wfa, f)

    try:
        ## The latest counter example
        # This exists because in WFA learning, it can occur that the already given counter example is found again due to the rank torelance.
        def assert_timeout():
            if params.time_limit is not None:
                if time() - lstar_start > params.time_limit:
                    raise LstarTimedOut()

        last_ce = None
        while True:
            start = time()
            table.make_table_closed_and_consistent(assert_timeout)
            wfa = table.reconstruct_WFA(assert_timeout)
            elapsed_table = time() - start
            table_time += elapsed_table
            add_stat()
            mylogger.info(f"Refinement of the table took {elapsed_table}")
            start = time()
            counterexample, stat_in_eqq = eqq.answer_query(wfa, table.S, assert_timeout)
            elapsed_eqq = time() - start
            eqq_time += elapsed_eqq
            mylogger.info(f"Making counterexample took {elapsed_eqq}")
            stats_in_eqq.append(stat_in_eqq)
            if isinstance(counterexample, equiv_query.ResultAnswerQuery.Equivalent):
                mylogger.info(f"The WFA and RNN seems equivalent!")
                break
            elif isinstance(counterexample, equiv_query.ResultAnswerQuery.Counterexample):
                start = time()
                res = table.add_counterexample(counterexample.content, assert_timeout)
                elapsed_add_ce_time = time() - start
                mylogger.info(f"Adding a coounterexample took {elapsed_add_ce_time}")
                add_ce_time += elapsed_add_ce_time
                if last_ce == counterexample.content:
                    mylogger.info(
                        f"A counterexample {last_ce} is given twice.  "
                        + f"RNN value is {target.get_value(last_ce)} and WFA value is {wfa.get_value(last_ce)}")
                    new_tol = table.shrink_tol_rank()
                    # wfa = table.reconstruct_WFA()
                    mylogger.info(f"tol_rank is shrinked to {new_tol}.")
                    last_ce = None
                else:
                    # table.add_counterexample(counterexample.content, assert_timeout)
                    # wfa = table.reconstruct_WFA()
                    mylogger.debug(f"Got a counterexample {counterexample.content}")
                    last_ce = counterexample.content
            else:
                print(counterexample)
                assert False
            assert_timeout()
    except QuantitativeObservationTable.TableTimedOut:
        logging.warning("Stopped L* by TableTimedOut")
    except equiv_query.EquivalenceQueryTimedOut:
        logging.warning("Stopped L* by EquivalenceQueryTimedOut")
    except LstarTimedOut:
        logging.warning("Stopped L* by LstarTimedOut")
    except QuantitativeObservationTable.TooSmallRankTolerance:
        logging.warning("Too small rank tolerance.")
    except RNN.TooLongWord:
        logging.warning("Aborted by TooLongWord exception")
    except equiv_query.TooLongWordExceptionAndHalted as e:
        logging.warning(f"Aborted by TooLongWordExceptionAndHalted ({e.word})")
    except KeyboardInterrupt:
        logging.warning("Stopped L* by KeyboardInterrupt")
    if wfa is not None:
        add_stat()
        stat.extracting_time = time() - lstar_start
        stat.size_wfa = wfa.get_size()
        stat.eqq_time = eqq_time
        stat.table_time = table_time
        stat.add_ce_time = add_ce_time
        stat.stats_in_eqq = stats_in_eqq
        return LStarResult(wfa, stat)
    else:
        assert False
