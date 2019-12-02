import unittest
import Lstar
import QuantitativeObservationTable
import equiv_query_regr
import WFA
import numpy as np
import sklearn
import sklearn.gaussian_process


class TestFunctions(unittest.TestCase):
    def setUp(self) -> None:
        alpha = np.array([[1, 3, 4]])
        beta = np.array([[2, 1, 1]]).T
        ma = np.array([
            [0, 0, 3],
            [0, 0, 3],
            [1, 0, 0]
        ])
        mb = np.array([
            [0, 1, 0],
            [2, 0, 0],
            [0, 0, 4]
        ])

        wfa = WFA.WFA("ab", alpha, beta, {"a": ma, "b": mb})
        self.wfa = wfa

    def test_run_quantitative_lstar(self):
        params_q = QuantitativeObservationTable.QuantitativeObservationTableParameters(
            normalization=False,
            tol_rank_init=0.01,
            tol_rank_decay_rate=0.5,
            tol_rank_lower_bound=1e-8,
            time_limit=60 * 3
        )
        regressor_maker = lambda: sklearn.gaussian_process.GaussianProcessRegressor()
        params_e = equiv_query_regr.EquivalenceQueryParameters(
            e=0.1,
            eta=0.1,
            gamma=0.1,
            cap_m=10,
            depth_eager_search=3,
            regressor_maker=regressor_maker
        )
        params = Lstar.LstarParameters(
            params_qot=params_q,
            params_eqq=params_e
        )
        res = Lstar.run_quantitative_lstar(self.wfa, params)
        print(res.show_wfa())
        words = ["", "ab", "aaa", "abababab", "bbbbbbbbbbb"]
        for word in words:
            self.assertAlmostEqual(self.wfa.get_value(word), res.get_value(word))
