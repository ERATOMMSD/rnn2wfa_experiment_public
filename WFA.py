from typing import *
import numpy as np
import ContinuousStateMachine
# import cvxopt  # usage: https://cvxopt.org/userguide/coneprog.html


def calc_nearest_trans_mat(target: np.ndarray) -> np.ndarray:
    """
     Returns the nearest matrix in the sense of L2 distance whose ROWS' values are in [0, 1] and whose sums are all 1
    :param target: np.ndarray of size (n, n)
    :return: np.ndarray of size (n, n)
    """
    assert False
    # size = target.size
    # n = len(target)
    # q = (-2) * (target.reshape(-1, )).reshape((-1, 1))
    # P = 2 * np.eye(size)
    # G = (-1) * np.eye(size)
    # h = np.zeros(size).reshape((-1, 1))
    # A = np.kron(np.eye(n), np.ones(n).reshape((1, -1)))
    # b = np.ones(n).reshape((-1, 1))
    # from cvxopt import matrix
    # cvxopt.solvers.options['show_progress'] = False
    # sol = cvxopt.solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
    # solX = np.array(sol["x"])
    # return solX.reshape((n, n))


def calc_nearest_vec_of_sum1(target):
    """
    Returns the nearest vector in the sense of L2 distance whose values are in [0, 1] and whose sum is 1
    :param target: np.ndarray
    :return: np.ndarray of size (-1,)
    """
    assert False
    # size = target.size
    # q = (-2) * (target.reshape(-1, )).reshape((-1, 1))
    # P = 2 * np.eye(size)
    # # G and h are for the condition of 0 <= x
    # G = (-1) * np.eye(size)
    # h = np.zeros(size).reshape((-1, 1))
    # # A and b are for the condition of the sums of rows
    # A = np.ones(size).reshape((1, -1))
    # b = np.ones(1).reshape((-1, 1))
    # # the conditions above induces x <= 1, so there is no need of write x <= 1 explicitly
    # from cvxopt import matrix
    # cvxopt.solvers.options['show_progress'] = False
    # sol = cvxopt.solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
    # solX = np.array(sol["x"])
    # return solX.reshape((-1,))


def calc_nearest_vec_of_0to1(target):
    """
    Returns the nearest vector in the sense of L2 distance whose values are in [0, 1]
    :param target: np.ndarray
    :return: np.ndarray of size (-1,)
    """
    assert False
    # # assume column vector
    # size = target.size
    # q = (-2) * (target.reshape(-1, )).reshape((-1, 1))
    # P = 2 * np.eye(size)
    # # G1 and h1 are for the condition of 0 <= x
    # G1 = (-1) * np.eye(size)
    # h1 = np.zeros(size).reshape((-1, 1))
    # # G2 and h2 are for the condition of x <= 1
    # G2 = np.eye(size)
    # h2 = np.ones(size).reshape((-1, 1))
    # # G and h
    # G = np.vstack([G1, G2])
    # h = np.vstack([h1, h2])
    # from cvxopt import matrix
    # cvxopt.solvers.options['show_progress'] = False
    # sol = cvxopt.solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
    # solX = np.array(sol["x"])
    # return solX.reshape((-1,))


class WFA(ContinuousStateMachine.ContinuousStateMachine):
    def __init__(self,
                 alphabet: str,
                 q0: np.ndarray,
                 final: np.ndarray,
                 delta: Dict[str, np.ndarray]) -> None:
        """
        :param alphabet: listing of the alphabets as a string
        :param q0: matrix of size 1 x n
        :param final: matrix of size n x 1
        :param delta: matrices of size n x n
        """
        self.alphabet: str = alphabet
        self.q0: np.ndarray = q0
        self.final: np.ndarray = final
        self.delta: Dict[str, np.ndarray] = delta
        self.callings: Set[str] = set()

        # assertion
        assert len(self.alphabet) > 0
        n = self.q0.size
        assert n == self.final.size
        for k, v in self.delta.items():
            assert k in self.alphabet
            assert v.shape == (n, n)

    def classify_word(self,
                      word: str) -> float:
        """
        :param word: assumes word is string with only letters in alphabet
        :return: f_A(w)
        """
        self.callings.add(word)
        return self.calc_result(self.calc_states(word))

    def calc_states(self,
                    word: str) -> np.ndarray:
        """
        Get A's corresponding configuration for word.
        we assume word is string with only letters in alphabet
        :param word:
        :return: ¥delta_A(w)
        """
        return self.calc_next(self.q0, word)

    def calc_next(self,
                  state: np.ndarray,
                  word: str) -> np.ndarray:
        """
        Calculate the next configuration from the given configuration
        :param state:
        :param word:
        :return:
        """
        q = state
        for a in word:
            q = np.dot(q, self.delta[a])
        return q

    def calc_result(self,
                    state: np.ndarray) -> float:
        """
        Calculates $state ¥cdot final$
        :param state:
        :return:
        """
        return float(np.dot(state, self.final))

    def get_configuration(self, w: str) -> np.ndarray:
        return self.calc_states(w).reshape((1, -1))

    def get_value(self, w: str) -> float:
        return self.classify_word(w)

    def show_wfa(self) -> str:
        s = ""
        s += f"q0: {self.q0.reshape((-1,))}\n"
        s += f"final: {self.final.reshape((-1,))}\n"
        for a in self.alphabet:
            s += f"delta[{a}]\n"
            s += str(self.delta[a])
            s += "\n"
        return s

    def calc_average(self, length: int) -> float:
        n = len(self.alphabet)
        words = (1 - n ** (length + 1)) / (1 - n)
        m = sum(self.delta.values())
        msum = sum([np.linalg.matrix_power(m, i) for i in range(length + 1)])
        res = float(self.q0.dot(msum).dot(self.final)) / words
        return res

    def get_size(self) -> int:
        return self.q0.size

    def get_callings(self) -> int:
        return len(self.callings)


def normalize(wfa: WFA) -> WFA:
    alphabet = wfa.alphabet
    q0 = calc_nearest_vec_of_sum1(wfa.q0).reshape((1, -1))
    final = calc_nearest_vec_of_0to1(wfa.final).reshape((-1, 1))
    delta = {k: calc_nearest_trans_mat(v) for k, v in wfa.delta.items()}
    return WFA(alphabet, q0, final, delta)
