import argparse
import shutil
import zipfile
import math
import os
import pickle
from typing import *
import numpy as np
import progressbar
import tensorflow as tf
import WFA
import RNN
import Dataset
import util_boto3
import json
import glob
import time
import precalc_rnn
import RNN_ADFA


#
# class EvalprefResult(NamedTuple):
#     str2rnnvalue: Dict[str, float]
#     str2wfavalue: Dict[str, float]
#     time_rnn: float
#     time_wfa: float
# 
#     def get_dict(self) -> Dict[str, Any]:
#         return {"time_rnn": self.time_rnn,
#                 "str2rnnvalue": self.str2rnnvalue,
#                 "str2wfavalue": self.str2wfavalue,
#                 "time_wfa": self.time_wfa}


skip_upload = False


class WFAData(NamedTuple):
    str2wfavalue: Dict[str, float]
    time_wfa: float
    mse: float  # Mean Squared Error
    acc005: float

    def get_dict(self) -> Dict[str, Any]:
        return {"time_wfa": self.time_wfa,
                "mse": self.mse,
                "str2wfavalue": self.str2wfavalue,
                "acc005": self.acc005}


def calc_wfa_rnn(wfa: WFA.WFA, str2rnnvalue: Dict[str, float]) -> WFAData:
    sum_sq_error = 0
    str2wfavalue = {}
    num_correct005 = 0
    start = time.time()
    for k, v in str2rnnvalue.items():
        str2wfavalue[k] = wfa.get_value(k)
    time_wfa = time.time() - start
    for k in str2rnnvalue.keys():
        diff = str2wfavalue[k] - str2rnnvalue[k]
        sum_sq_error += diff ** 2
        if abs(diff) < 0.05:
            num_correct005 += 1
    return WFAData(str2wfavalue, time_wfa, sum_sq_error / len(str2rnnvalue), num_correct005 / len(str2rnnvalue))


adfa_words = {151: "adfa_test_151.zip",
              10: "adfa_test_10.zip",
              20: "adfa_test_20.zip"}


def chrword(xs):
    if xs == "":
        return ""
    return "".join([chr(int(i)) for i in xs.split(",")])


def calc_adfa(args_s3extracted, dirname):
    splitted = args_s3extracted[:-4].split("_")
    proj = splitted[-2]
    alph = int(splitted[-1])
    wordfile = adfa_words[alph]
    util_boto3.download(wordfile)
    with zipfile.ZipFile(wordfile) as z:
        z.extractall(dirname)
        try:
            shutil.copy(os.path.join(dirname, wordfile[:-4], "test.txt"), dirname)
        except FileNotFoundError:
            pass
    with open(os.path.join(dirname, "test.txt"), 'r') as f:
        words = [w.strip() for w in f]
        words = [chrword(x) for x in words]
    csm = RNN_ADFA.RNNAdfa(proj, alph)
    print("Starting evaluation")
    res = precalc_rnn.evaluate_csm(csm, words)
    print("Finished evaluation")
    return res.get_dict()


def calc_paren(args_s3extracted, dirname):
    alph = "()0123456789"
    alphabet2id, max_length, regr, words = precalc_rnn.set_for_paren(args_s3extracted, dirname)
    print("Starting evaluation")
    res = precalc_rnn.evaluate(regr, alphabet2id, words, dirname)
    print("Finished evaluation")
    return res.get_dict()


def run(args_s3extracted, args_s3precalc, args_s3):
    dirname = "temp_evalperf"
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)

    if "adfa" in args_s3extracted:
        flag_adfa = True
    else:
        flag_adfa = False
    flag_paren = "paren" in args_s3extracted


    # download
    util_boto3.download(args_s3extracted)
    with zipfile.ZipFile(args_s3extracted) as z:
        z.extractall(dirname)
    if not flag_adfa and not flag_paren:
        util_boto3.download(args_s3precalc)
        with zipfile.ZipFile(args_s3precalc) as z:
            z.extractall(dirname)

    # load wfa
    with open(os.path.join(dirname, "wfa_extracted.pickle"), "rb") as f:
        wfa: WFA.WFA = pickle.load(f)
    with open(os.path.join(dirname, "statistics.json"), "r") as f:
        stat = json.load(f)
        periods_lstar = stat["periods_lstar"]
    if flag_adfa:
        precalc = calc_adfa(args_s3extracted, dirname)
    elif flag_paren:
        precalc = calc_paren(args_s3extracted, dirname)
    else:
        with open(os.path.join(dirname, "result_precalc.json"), "r") as f:
            precalc = json.load(f)
    str2rnnvalue = precalc["str2rnnvalue"]
    time_rnn = precalc["time_rnn"]
    with open(os.path.join(dirname, "args_rnn2wfa.json"), "r") as f:
        a_rnn2wfa = json.load(f)
    if not flag_adfa and not flag_paren:
        with open(os.path.join(dirname, "args_precalc.json"), "r") as f:
            a_precalc = json.load(f)
        assert a_rnn2wfa["s3rnn"] == a_precalc["s3rnn"]

    wfa_data = calc_wfa_rnn(wfa, str2rnnvalue)
    process_data = []
    if a_rnn2wfa["save_process"]:

        for i, _ in enumerate(periods_lstar):
            with open(os.path.join(dirname, f"wfa{i}.pickle"), "rb") as f:
                wfap: WFA.WFA = pickle.load(f)
                process_data.append(calc_wfa_rnn(wfap, str2rnnvalue))

    # write result
    # print(str2rnnvalue)
    str2rnnvalue = {k: float(v) for k, v in str2rnnvalue.items()}
    d = {"wfa_data": wfa_data.get_dict(), "process_data": [x.get_dict() for x in process_data], "time_rnn": time_rnn,
         "str2rnnvalue": str2rnnvalue}
    with open("result_eval_perf.json", "w") as f:
        json.dump(d, f)
    # write args
    with open("args_eval_perf.json", "w") as f:
        d = {"s3extracted": args_s3extracted, "s3": args_s3, "s3precalc": args_s3precalc}
        json.dump(d, f)

    # zip and upload
    z = util_boto3.zip_and_upload(args_s3,
                                  ["result_eval_perf.json", "args_eval_perf.json"] + glob.glob(
                                      os.path.join(dirname, "*")), True,
                                  skip_upload=skip_upload)
    print(z)
    return z


def main():
    parser = argparse.ArgumentParser(
        description='Calc and count')

    parser.add_argument('--s3', action="store_true",
                        help='upload to s3')
    parser.add_argument('-o', default="result_eval_perf.zip",
                        help='filename of s3')
    parser.add_argument('extracted', default="",
                        help='extracted wfa.  If it is not found, downloaded from s3.')
    parser.add_argument('precalc', default="",
                        help='the result of "precalc".  If it is not found, downloaded from s3.')
    args = parser.parse_args()

    global skip_upload
    skip_upload = args.s3

    run(args_s3extracted=args.extracted,
        args_s3precalc=args.precalc,
        args_s3=args.o)


if __name__ == '__main__':
    main()
