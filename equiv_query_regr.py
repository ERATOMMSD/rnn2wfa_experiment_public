import numpy as np
import logging
from typing import *
import functools
import WFA
import ContinuousStateMachine
import util
import itertools
from typing_extensions import Protocol
from time import time
import equiv_query
import sklearn
import sklearn.gaussian_process
import sklearn.kernel_ridge
import sklearn.ensemble.gradient_boosting
import sklearn.neural_network
import math
import preserving_heapq
import RNN

mylogger = logging.getLogger("rnn2wfa").getChild("equiv_query_regr")


def infnorm(x: np.ndarray) -> float:
    return np.linalg.norm(x, ord=np.inf)


class SupportRegression(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...



class EquivalenceQueryParameters:
    e: float
    eta: float
    # gamma: float
    cap_m: int
    depth_eager_search: int
    regressor_maker: Callable[[], SupportRegression]
    time_limit: Optional[int]
    experimental_constant_allowance: bool
    experimental_automatic_eta: bool
    experimental_disable_restarting: bool
    experimental_learn_weiss_style: bool
    experimental_bfs: bool
    experimental_sort_by_dist: bool
    experimental_disable_equivalent: bool
    experimental_skip_existing: bool
    experimental_use_next_value: bool
    experimental_halt_long_word: Optional[int]
    experimental_wfa_only: bool

    def __init__(self, comment: str, eps: float, max_length: int,
                 eta: float, gamma: float, cap_m: int,
                 depth_eager_search: int, regressor_maker: Optional[Callable[[], SupportRegression]] = None,
                 regressor_maker_name: str = "",
                 time_limit: Optional[int] = None,
                 experimental_constant_allowance=False,
                 experimental_automatic_eta=False,
                 experimental_disable_restarting=False,
                 experimental_learn_weiss_style=False,
                 experimental_bfs=False,
                 experimental_sort_by_dist=False,
                 experimental_disable_equivalent=False,
                 experimental_skip_existing=False,
                 experimental_use_next_value=False,
                 experimental_halt_long_word=None,
                 experimental_wfa_only=False):
        """

        :param eps:
        :param max_length:
        :param eta:
        :param cap_m:
        :param depth_eager_search:
        :param regressor_maker:
        :param regressor_maker_name:
        :param time_limit:
        :param experimental_constant_allowance:
        """
        self.comment: str = comment
        self.e: float = eps if experimental_constant_allowance else eps / max_length
        self.eta: float = eta
        # self.gamma: float = gamma
        self.cap_m: int = cap_m
        self.depth_eager_search = depth_eager_search
        self.experimental_constant_allowance = experimental_constant_allowance
        self.experimental_automatic_eta = experimental_automatic_eta
        self.experimental_disable_restarting = experimental_disable_restarting
        self.experimental_learn_weiss_style = experimental_learn_weiss_style
        self.experimental_bfs = experimental_bfs
        self.experimental_sort_by_dist = experimental_sort_by_dist
        self.experimental_disable_equivalent = experimental_disable_equivalent
        self.experimental_skip_existing = experimental_skip_existing
        self.experimental_use_next_value = experimental_use_next_value
        self.experimental_halt_long_word = experimental_halt_long_word
        self.experimental_wfa_only = experimental_wfa_only
        if regressor_maker is not None:
            self.regressor_maker = regressor_maker
        else:
            if regressor_maker_name == "gaussian" or regressor_maker_name == "gpr":
                self.regressor_maker = lambda: sklearn.gaussian_process.GaussianProcessRegressor()
            elif regressor_maker_name == "krr":
                self.regressor_maker = lambda: sklearn.kernel_ridge.KernelRidge()
            elif regressor_maker_name == "dtr":
                self.regressor_maker = lambda: sklearn.ensemble.gradient_boosting.DecisionTreeRegressor()
            # elif regressor_maker_name == "mlpr":
            #     self.regressor_maker = lambda: sklearn.neural_network.MLPRegressor()
            else:
                assert False
        self.time_limit: Optional[int] = time_limit


class EquivalenceQueryAnswerer(equiv_query.EquivalenceQueryAnswererBase):
    samples: Set[str]
    start: float

    def __init__(self, rnn: ContinuousStateMachine.ContinuousStateMachine,
                 params: EquivalenceQueryParameters,
                 dirname: str):
        self.rnn: ContinuousStateMachine.ContinuousStateMachine = rnn
        self.params = params
        self.regressor: SupportRegression = self.params.regressor_maker()

    def _reset_timeout(self):
        self.start = time()

    def _assert_not_timeout(self):
        if self.params.time_limit is not None:
            if time() - self.start > self.params.time_limit:
                raise equiv_query.EquivalenceQueryTimedOut()

    def is_around_in_wfa_config(self, x: np.ndarray, y: np.ndarray, wfa: WFA.WFA) -> bool:
        if self.params.experimental_automatic_eta:
            return util.dist_f(lambda a, b: np.linalg.norm(a - b),
                               lambda t: np.reshape(t, (-1,)) * np.reshape(wfa.final, (-1,)),
                               x, y) < self.params.e / math.sqrt(wfa.get_size())
        else:
            assert False
            # Naive style is not used recently and disabled.
            # return util.dist_f(lambda a, b: infnorm(a - b), lambda x: x, x, y) < self.params.eta

    def _update_p_and_get_p_delta_r(self, visited: Iterable[str], wfa: WFA.WFA) -> Callable[[str], np.ndarray]:
        points_rnn = []
        points_wfa = []
        for v in visited:
            points_rnn.append(self.rnn.get_configuration(v))
            points_wfa.append(wfa.get_configuration(v))
        points_rnn_stacked = np.vstack(points_rnn)
        points_wfa_stacked = np.vstack(points_wfa)
        # print(points_rnn, points_wfa, points_rnn_stacked, points_wfa_stacked)
        self.regressor.fit(points_rnn_stacked, points_wfa_stacked)

        @functools.lru_cache(maxsize=None)
        def p_delta_r(h: str) -> np.ndarray:
            return self.regressor.predict(self.rnn.get_configuration(h))

        return p_delta_r

    def _get_criteria_string_difference(self, s: str):
        if self.params.experimental_constant_allowance:
            return self.params.e
        else:
            assert False
            # Not used these days
            # return self.params.e * (len(s) + 1)

    def answer_query(self, wfa: WFA.WFA, existing_ces: Iterable[str], assert_timeout: Callable[[], None]) -> Tuple[
        equiv_query.ResultAnswerQuery.T, Any]:
        mylogger.info("Starting answer_query")
        if self.params.experimental_bfs:
            mylogger.info("BFS is available.")
        self._reset_timeout()
        learn_samples = [""]
        p_delta_r = None if self.params.experimental_wfa_only else self._update_p_and_get_p_delta_r([""], wfa)
        time_regression = 0
        time_finding_points = 0
        time_calc_min = 0

        def make_info():
            info = {"time_regression": time_regression,
                    "time_finding_points": time_finding_points,
                    "time_calc_min": time_calc_min}
            return info

        # self.regressor.fit(self.rnn.get_configuration(""), self.wfa.calc_states(""))
        while True:
            self._assert_not_timeout()
            restart = False
            queue: preserving_heapq.PreservingHeapQueue[str, float] = preserving_heapq.PreservingHeapQueue()
            if self.params.experimental_sort_by_dist:
                queue.push_with_priority("", -np.inf)
            else:
                queue.push_with_priority("", 0)
            visited = []
            while not queue.is_empty():
                # mylogger.debug(f"Queue: {queue.queue}")
                assert_timeout()
                h, n = queue.pop()
                self.assert_popped(n)
                if self.params.experimental_halt_long_word is not None \
                        and len(h) > self.params.experimental_halt_long_word:
                    raise equiv_query.TooLongWordExceptionAndHalted(h)
                try:
                    if abs(self.rnn.get_value(h) - wfa.classify_word(h)) >= self._get_criteria_string_difference(h):
                        mylogger.debug(f"Found a counterexample '{h}' in usual search")
                        if self.params.experimental_skip_existing and h in existing_ces:
                            mylogger.debug(f"Skipping counterexample '{h}'")
                        else:

                            return equiv_query.ResultAnswerQuery.Counterexample(h), make_info()
                except RNN.TooLongWord:
                    continue
                if self.params.experimental_wfa_only == False:
                    result = self.is_consistent(h, visited, p_delta_r, wfa, assert_timeout)
                    if isinstance(result, equiv_query.ResultIsConsistent.NG):
                        time_regression = self.update_regressor(h, learn_samples, time_regression, visited, wfa)
                        if self.params.experimental_disable_restarting:
                            pass
                        else:
                            restart = True
                            break
                    elif isinstance(result, equiv_query.ResultIsConsistent.Counterexample):
                        raise Exception("Counterexample generation in consistency check is currently disabled.")
                        mylogger.info(f"Found a counterexample in consistency check: '{result}")
                        return equiv_query.ResultAnswerQuery.Counterexample(result.content), make_info
                    elif isinstance(result, equiv_query.ResultIsConsistent.OK):
                        pass
                    else:
                        assert False
                visited.append(h)

                point_getter = wfa.get_configuration if self.params.experimental_wfa_only else p_delta_r
                time_calc_min, time_finding_points, pruned = self.proceed_bfs(h, point_getter, queue, time_calc_min,
                                                                              time_finding_points, visited, wfa)
            if not restart:
                break
        mylogger.info("Seems equivalent")
        return equiv_query.ResultAnswerQuery.Equivalent(), make_info()

    def assert_popped(self, n: Union[float, int]) -> None:
        if self.params.experimental_bfs:
            if self.params.experimental_sort_by_dist:
                assert n <= 0
            else:
                assert n >= 0
        else:
            assert n == 0

    def update_regressor(self, h: str, learn_samples: List[str], time_regression: float, visited: List[str],
                         wfa: WFA.WFA) -> float:
        learn_samples.append(h)
        time_temp = time()
        if self.params.experimental_learn_weiss_style:
            self._update_p_and_get_p_delta_r(learn_samples, wfa)
        else:
            self._update_p_and_get_p_delta_r(visited + [h], wfa)
        time_regression += time() - time_temp
        mylogger.info(f"updated p at {h}")
        return time_regression

    def proceed_bfs(self, h: str, p_delta_r: Callable[[str], np.ndarray], queue: preserving_heapq.PreservingHeapQueue,
                    time_calc_min: float, time_finding_points: float, visited: List[str], wfa: WFA.WFA) -> Tuple[
        float, float, bool]:
        time_temp = time()

        if self.params.experimental_disable_equivalent:
            if self.params.experimental_bfs:
                if self.params.experimental_sort_by_dist:
                    flag_calc_pfrh_omitted = True
                else:
                    if self.params.experimental_use_next_value:
                        flag_calc_pfrh_omitted = True
                    else:
                        flag_calc_pfrh_omitted = False
            else:
                flag_calc_pfrh_omitted = True
        else:
            flag_calc_pfrh_omitted = False

        if flag_calc_pfrh_omitted:
            points_around_pfrh = []
        else:
            points_around_pfrh = [h1 for h1 in visited if
                                  self.is_around_in_wfa_config(p_delta_r(h), p_delta_r(h1), wfa)]
            flag_calc_pfrh_omitted = False
        time_finding_points += time() - time_temp
        if self.params.experimental_disable_equivalent or len(points_around_pfrh) <= self.params.cap_m:
            pruned = False
            for sigma in self.rnn.alphabet:
                h_next = h + sigma
                if self.params.experimental_bfs:
                    if self.params.experimental_sort_by_dist:
                        if self.params.experimental_use_next_value:
                            dist_min, time_calc_min = self.calc_dist_min(h_next, p_delta_r, time_calc_min, visited)
                        else:
                            dist_min, time_calc_min = self.calc_dist_min(h, p_delta_r, time_calc_min, visited)
                        # if dist_min is small, it seems dense around p_delta_r(h)
                        queue.push_with_priority(h_next, -dist_min)
                    else:
                        if self.params.experimental_use_next_value:
                            points_around_pfrh_next = [h1 for h1 in visited if
                                                       self.is_around_in_wfa_config(p_delta_r(h_next), p_delta_r(h1),
                                                                                    wfa)]
                            around_num = len(points_around_pfrh_next)
                        else:
                            assert not flag_calc_pfrh_omitted
                            around_num = len(points_around_pfrh)
                        queue.push_with_priority(h_next, around_num)
                else:
                    queue.push_with_priority(h_next, 0)
        else:
            pruned = True
            mylogger.info(f"Pruning the branch of '{h}'.  The remaining branches are {len(queue)}.")
        return time_calc_min, time_finding_points, pruned

    def calc_dist_min(self, h: str, p_delta_r: Callable[[str], np.ndarray], time_calc_min: float, visited: List[str]) -> \
            Tuple[float, float]:
        time_temp = time()
        calc_min_list = [np.linalg.norm(p_delta_r(h) - p_delta_r(h1)) for h1 in visited if h1 != h]
        if calc_min_list:
            dist_min = min(calc_min_list)
        else:
            dist_min = np.inf
        time_calc_min += time() - time_temp
        return dist_min, time_calc_min

    def _s_generator(self, wfa: WFA.WFA, h: str, h1: str, cap_x: List[str]) -> Iterator[str]:
        for s in util.bfs_words(self.rnn.alphabet, self.params.depth_eager_search):
            if abs(wfa.classify_word(h1 + s) - wfa.classify_word((h + s))) \
                    >= self._get_criteria_string_difference(h1 + s):
                h2_iter = (h2 for h2 in itertools.chain(cap_x, [h])
                           if
                           abs(wfa.classify_word(h2 + s) - self.rnn.get_value(h2 + s))
                           >= self._get_criteria_string_difference(h2 + s))
                try:
                    h2 = next(h2_iter)
                    yield h2 + s
                except StopIteration:
                    # could not find proper h2
                    return

    def is_consistent(self,
                      h: str,
                      visited: Iterable[str],
                      p_delta_r: Callable[[str], np.ndarray],
                      wfa: WFA.WFA,
                      assert_timeout: Callable[[], None]) -> equiv_query.ResultIsConsistent.T:
        cap_x = [h1 for h1 in visited if self.is_around_in_wfa_config(p_delta_r(h1), p_delta_r(h), wfa)]
        if not cap_x:
            mylogger.info("p seems consisten because X was empty")
            return equiv_query.ResultIsConsistent.OK()
        else:
            h1_iter = (h1 for h1 in cap_x if
                       not self.is_around_in_wfa_config(p_delta_r(h1), wfa.get_configuration(h1), wfa))
            try:
                h1 = next(h1_iter)
            except StopIteration:
                # cap_x is empty
                mylogger.info("p seems consistent because it is consistent to old information")
                return equiv_query.ResultIsConsistent.OK()

            if self.params.depth_eager_search < 0:
                mylogger.info("p seems wrong")
                return equiv_query.ResultIsConsistent.NG()
            else:
                assert False
                # Not used these days and disabled
                # s_iter = self._s_generator(wfa, h, h1, cap_x)
                # try:
                #     counterexample = next(s_iter)
                #     return equiv_query.ResultIsConsistent.Counterexample(counterexample)
                # except StopIteration:
                #     # could not found counterexample
                #     return equiv_query.ResultIsConsistent.NG()
