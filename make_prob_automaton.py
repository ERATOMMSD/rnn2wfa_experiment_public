import argparse
import os
import pickle
import random
from typing import *
import numpy as np
from WFA import WFA
import util_boto3
import json


def make_distr(n_cells: int):
    s = n_cells * n_cells
    x = list("o" * s + "|" * (n_cells - 1))
    random.shuffle(x)
    x = "".join(x).split("|")
    return [len(i) / s for i in x]


def make_trans_vec(n_states, deg):
    d = make_distr(deg)
    res = [0.] * (n_states - deg) + d
    random.shuffle(res)
    return np.array(res)


def make_trans_mat(n_states, deg):
    return np.array([make_trans_vec(n_states, deg) for _ in range(n_states)])


def make_prob_automaton(alphabets: str,
                        n_states: int,
                        deg: int,
                        final_dist: Callable[[], float]) -> WFA:
    q0 = np.array(make_trans_vec(n_states, deg)).reshape(1, -1)
    # TODO: Should not final vectors be normalized?a
    final = np.array([final_dist() for _ in range(n_states)]).reshape(-1, 1)
    trans = {a: np.array(make_trans_mat(n_states, deg)) for a in alphabets}
    return WFA(alphabets, q0, final, trans)


def main():
    parser = argparse.ArgumentParser(
        description='Make a probabilistic automaton')
    parser.add_argument('alphabets',
                        help='The alphabet sets')
    parser.add_argument('n_states', type=int,
                        help='The number of states')
    parser.add_argument('degree', type=int,
                        help='The degree of each state')
    parser.add_argument("--s3", action="store_true",
                        help="upload to S3")
    parser.add_argument("-o", type=str, default="result_make_prob_automaton.zip",
                        help="filename of output")
    args = parser.parse_args()

    wfa = make_prob_automaton(args.alphabets,
                              args.n_states,
                              args.degree,
                              (lambda: np.random.beta(0.5, 0.5)))

    with open("wfa.pickle", 'wb') as f:
        pickle.dump(wfa, f)
    with open("args.json", "w") as f:
        json.dump(vars(args), f)

    fn = util_boto3.zip_save(args.o, ["wfa.pickle", "args.json"], True)
    if args.s3:
        util_boto3.upload(fn)
        print("uploaded")
    print(fn)


if __name__ == '__main__':
    main()
