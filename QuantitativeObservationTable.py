# -*- coding: utf-8 -*-

from time import time
from typing import *
import numpy as np
import logging
import WFA

mylogger = logging.getLogger("rnn2wfa").getChild("quantitative_observation_table")


class TableTimedOut(Exception):
    pass


class TooSmallRankTolerance(Exception):
    pass


class QuantitativeObservationTableParameters:
    normalization: bool
    tol_rank_init: float
    tol_rank_decay_rate: float
    tol_rank_lower_bound: float
    time_limit: Optional[int]

    def __init__(self, normalization: bool,
                 tol_rank_init: float,
                 tol_rank_decay_rate: float,
                 tol_rank_lower_bound: float,
                 time_limit: Optional[int]):
        self.normalization = normalization
        self.tol_rank_init = tol_rank_init
        assert tol_rank_init > 0
        self.tol_rank_decay_rate = tol_rank_decay_rate
        assert tol_rank_decay_rate > 0
        self.tol_rank_lower_bound = tol_rank_lower_bound
        assert tol_rank_lower_bound >= 0
        assert tol_rank_init > tol_rank_lower_bound
        self.time_limit = time_limit


class QuantitativeObservationTable:
    """
    All the parameters are stored in QuantitativeObservationTableParameters
    Memoization of the membership-query is not our business, so self.T is deleted.
    Adjusting the tolerance or remembering the last CE is not our business neither.  It is done in the learning loop.
    """
    S: Set[str]
    E: Set[str]
    A: str
    start: float
    H: np.ndarray  # The above part of the learned table
    Ha: np.ndarray  # The below part o the learned table
    parameters: QuantitativeObservationTableParameters
    membership_query: Callable[[str], float]
    tol_rank: float

    def __init__(self,
                 alphabet: str,
                 membership_query: Callable[[str], float],
                 parameters: QuantitativeObservationTableParameters) -> None:
        self.S = {""}  # starts. invariant: prefix closed
        self.E = {""}  # ends. invariant: suffix closed
        self.A = alphabet  # alphabet
        self.membership_query = membership_query
        self._construct_hankel_matrix(lambda: None)
        self.parameters = parameters
        self.tol_rank = self.parameters.tol_rank_init

        if self.parameters.normalization:
            raise Exception(
                "Using normalization is now suspended.  See https://github.com/ERATOMMSD/rnn2wfa_experiment/issues/2")

    def call_membership_query(self, w: str, assert_timeout: Callable[[], None]) -> float:
        assert_timeout()
        return self.membership_query(w)

    def reset_timeout(self):
        self.start = time()

    def _SdotA(self) -> Set[str]:  # doesn't modify
        return set([s + a for s in self.S for a in self.A])

    def _construct_hankel_matrix(self, assert_timeout: Callable[[], None]) -> None:
        """
        This function just fill the Hankel sub-matrices H and Ha.
        """
        self.H = np.asarray([[self.call_membership_query(s + e, assert_timeout) for e in self.E] for s in self.S],
                            dtype=np.float64)
        self.Ha = np.asarray(
            [[self.call_membership_query(s + e, assert_timeout) for e in self.E] for s in self._SdotA()],
            dtype=np.float64)

    def _update_hankel_matrix(self,
                              new_e: Optional[str] = None,
                              new_s: Optional[str] = None,
                              assert_timeout: Callable[[], None] = None) -> None:
        # just fixes cache. in case of new_e - only makes it smaller
        if assert_timeout is None:
            assert_timeout = lambda: None
        if new_e is not None:
            # new_e and E must be disjoint
            self.H = np.concatenate(
                (self.H,
                 np.asarray([[self.call_membership_query(s + e, assert_timeout) for e in new_e] for s in self.S])),
                axis=1)
            self.Ha = np.concatenate(
                (self.Ha, np.asarray(
                    [[self.call_membership_query(s + e, assert_timeout) for e in new_e] for s in self._SdotA()])),
                axis=1)
        else:  # new_s != None, or a bug!
            if new_s is None:
                raise Exception("new_s != None, or a bug!")  # added for typing
            self._construct_hankel_matrix(assert_timeout)

    def reconstruct_WFA(self, assert_timeout: Optional[Callable[[], None]] = None) -> WFA.WFA:
        if assert_timeout is None:
            assert_timeout = lambda: None
        H = self.H  # size of (P, S)
        U, D_, V_ = np.linalg.svd(H, False)  # size of (P, k), (k), (k, S)
        k = (D_ > self.tol_rank).sum()  # number of major singular values
        if k == 0:
            return WFA.WFA(self.A, np.array([[0]]), np.array([[1]]), {a: np.array([[0]]) for a in self.A})
        D = np.diag(D_)  # (k, k)
        V = V_.transpose()  # (S, k), portrait
        # cut minor singular values
        U = U[:, :k]
        D = D[:k, :k]
        V = V[:, :k]
        # calc main part
        P = U.dot(D)  # size of (P, k), portrait
        S = V  # size of (S, k), portrait
        p = P.shape[0]
        s = S.shape[0]
        P_pinv = np.linalg.pinv(P)  # (k, P)
        S_pinv = np.linalg.pinv(S)  # (k, S)
        alpha = (P.T).dot(np.eye(p)[:, 0:1])
        beta = (S.T).dot(np.eye(s)[:, 0:1])
        alpha.resize((k,))
        beta.resize((k,))
        delta: Dict[str, np.ndarray] = {}
        for a in self.A:
            Ha: np.ndarray = np.asarray(
                [[self.call_membership_query(s + a + e, assert_timeout) for e in self.E] for s in self.S],
                dtype=np.float64)
            delta[a] = P_pinv.dot(Ha).dot(S_pinv.T)
        res = WFA.WFA(self.A, alpha, beta, delta)
        if self.parameters.normalization:
            res = WFA.normalize(res)
        return res

    def _assert_not_timed_out(self) -> None:
        if self.parameters.time_limit is not None:
            if time() - self.start > self.parameters.time_limit:
                mylogger.warning("obs table timed out")
                raise TableTimedOut()  # whatever, can't be bothered rn

    def find_and_handle_inconsistency(self, assert_timeout: Optional[
        Callable[[], None]] = None) -> bool:  # modifies - and whenever it does, calls _fill_T
        """
        When rank(H) != rank([H Ha]), it finds a ∈ A and e1 ∈ E such that
         adding a ・ e1 to E changes the rank of the Hankel sub-matrix and add it to E.
        :return: The Boolean value of (rank(H) != rank([H Ha]))
        """
        if assert_timeout is None:
            assert_timeout = lambda: None
        # returns whether table was inconsistent
        mylogger.debug("find_and_handle_inconsistency")
        # The elements of the "Ha" in [Balle and Mohri]
        mylogger.debug("making hankel matrix")
        HaE = np.asarray(
            [[self.call_membership_query(s + a + e, assert_timeout) for e in self.E for a in self.A] for s in self.S],
            dtype=np.float64)
        # The rank of the Hankel sub-matrix H
        rank = np.linalg.matrix_rank(self.H, tol=self.tol_rank)
        # If the rank of [H Ha] is same as that of H, it is closed.
        if rank == np.linalg.matrix_rank(np.concatenate((self.H, HaE), axis=1), tol=self.tol_rank):
            return False
        # find a ∈ A and e1 ∈ E such that adding a ・ e1 to E changes the rank of the Hankel sub-matrix
        ae1 = next((a + e1 for e1 in self.E for a in self.A if not rank == np.linalg.matrix_rank(
            np.concatenate(
                (self.H, np.asarray([self.call_membership_query(s + a + e1, assert_timeout) for s in self.S]).reshape(
                    len(self.S), 1)),
                axis=1),
            tol=self.tol_rank)), None)
        mylogger.debug("making hankel matrix END")
        if ae1 is None:
            return False
        # if ae1 in self.E:  # TODO: Very adhoc fix
        #     mylogger.warning("adhoc!")
        #     return False
        self.E.add(ae1)
        mylogger.debug("reconstruction starting")
        self._construct_hankel_matrix(
            assert_timeout)  # TODO: partial reconstruction does not work. It should be more efficient.
        mylogger.debug("reconstruction end")
        self._assert_not_timed_out()
        return True

    def find_and_close_row(self, assert_timeout: Callable[
        [], None]) -> bool:  # modifies - and whenever it does, calls _fill_T
        # returns whether table was unclosed
        rank = np.linalg.matrix_rank(self.H, tol=self.tol_rank)
        mylogger.debug("rank" + str(rank))
        if rank == np.linalg.matrix_rank(np.concatenate((self.H, self.Ha), axis=0), tol=self.tol_rank):
            return False
        hoge2 = [
            (self.H.shape, np.asarray([self.call_membership_query(s1 + a + e, assert_timeout) for e in self.E]).shape)
            for a in self.A
            for e in self.E
            for s1 in self.S]
        mylogger.debug("s1a", hoge2)
        s1a = next((s1 + a for s1 in self.S for a in self.A if not rank == np.linalg.matrix_rank(
            np.concatenate((self.H, np.asarray(
                [self.call_membership_query(s1 + a + e, assert_timeout) for e in self.E]).reshape((1, -1))),
                           axis=0),
            tol=self.tol_rank)), None)
        if s1a is None:
            return False
        self.S.add(s1a)
        self._update_hankel_matrix(new_s=s1a)
        self._assert_not_timed_out()
        return True

    def _add_counterexample_help(self,
                                 ce: str,
                                 assert_timeout: Callable[[], None]) -> bool:  # modifies - and definitely calls _fill_T
        if ce in self.S:
            mylogger.debug("bad counterexample - already saved and classified in table!")
            return False

        new_states = [ce[0:i + 1] for i in range(len(ce)) if not ce[0:i + 1] in self.S]

        for new_state in new_states:
            self.S.add(new_state)
            assert_timeout()
            self._construct_hankel_matrix(assert_timeout)
            ## We do not care the minimality of the Hankel sub-matrix!!
            # if not self.find_and_handle_inconsistency():
            #     self.S.remove(new_state)
            #     self._construct_hankel_matrix()
        self._assert_not_timed_out()

        mylogger.debug(f"added CE to the table {self.S} {self.E}")
        return True

    def add_counterexample(self,
                           ce: str,
                           assert_timeout: Optional[
                               Callable[[], None]] = None) -> bool:  # modifies - and definitely calls _fill_T
        if assert_timeout is None:
            assert_timeout = lambda: None
        res = self._add_counterexample_help(ce, assert_timeout)
        self._construct_hankel_matrix(assert_timeout)
        return res

    def make_table_closed_and_consistent(self, assert_timeout: Callable[[], None]):
        self.reset_timeout()
        while True:
            while self.find_and_handle_inconsistency(assert_timeout):
                # Make the table consistent while it is inconsistent
                pass
            if self.find_and_close_row(assert_timeout):
                # the table is unclosed
                continue
            else:
                # the table is closed and consistent
                break

    def shrink_tol_rank(self) -> float:
        """
        Return True if the current tol_rank is valid
        :return:
        """
        self.tol_rank *= self.parameters.tol_rank_decay_rate
        if self.tol_rank < self.parameters.tol_rank_lower_bound:
            raise TooSmallRankTolerance()
        return self.tol_rank

# Local Variables:
# flycheck-checker: python-mypy
# eval: (setq flycheck-python-mypy-ini (concat default-directory "mypy.ini"))
# End:
