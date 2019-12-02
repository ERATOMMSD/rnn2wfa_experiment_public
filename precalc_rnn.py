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
import ContinuousStateMachine


skip_upload = False


class PrecalcResult(NamedTuple):
    str2rnnvalue: Dict[str, float]
    time_rnn: float

    def get_dict(self) -> Dict[str, Any]:
        return {"time_rnn": self.time_rnn,
                "str2rnnvalue": self.str2rnnvalue}



def evaluate(regression: RNN.RNNRegression,
             alphabet2id: Dict[str, int],
             words: List[str],
             checkpoint: str) -> PrecalcResult:

    regr = regression
    csm = RNN.ContinuousStateMachine(regr, None, alphabet2id)
    saver = tf.train.Saver(max_to_keep=0)

    d = {}
    with tf.Session() as sess:
        if os.path.isdir(checkpoint):
            checkpoint = tf.train.latest_checkpoint(checkpoint)
        else:
            assert False

        csm._sess = sess
        saver.restore(sess, checkpoint)
        start = time.time()
        cnt = 0
        for w in words:
            d[w] = csm.get_value(w)
            cnt += 1
            print(f"{cnt}/{len(words)}")
        time_rnn = time.time() - start

    res = PrecalcResult(d, time_rnn)
    print(res)
            
    
    return res


def evaluate_csm(csm: ContinuousStateMachine.ContinuousStateMachine,
                 words: List[str]) -> PrecalcResult:
    d = {}
    start = time.time()
    for w in words:
        d[w] = csm.get_value(w)
    time_rnn = time.time() - start

    res = PrecalcResult(d, time_rnn)
    print(res)

    return res


masaki_tail = ["1-1.zip", "2-2.zip", "3-4.zip", "4-5.zip"]

def run(args_s3rnn, args_s3, paren=False):
    tf.reset_default_graph()
    dirname = "temp_precalc"
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)

    flag_masaki = any(args_s3rnn.endswith(i) for i in masaki_tail)

    # download
    util_boto3.download(args_s3rnn)
    if paren:
        alphabet2id, args_max_length, regr, words = set_for_paren(args_s3rnn, dirname)
        print(words)
    elif flag_masaki:
        alphabet2id, args_max_length, regr, words = set_for_masaki(args_s3rnn, dirname)
    else:
        alphabet2id, args_max_length, regr, words = set_for_artificial(args_s3rnn, dirname)

    # eval
    print("-" * 30 + "TEST" + "-" * 30)
    data_dummy = [(w, Dataset.word2vec(w, alphabet2id, args_max_length), 0) for w in words]
    res_test = evaluate(regr, alphabet2id, words, dirname)

    # write test result
    with open("result_precalc.json", "w") as f:
        json.dump(res_test.get_dict(), f)
    # write args
    with open("args_precalc.json", "w") as f:
        d = {"s3rnn": args_s3rnn, "s3": args_s3}
        json.dump(d, f)

    # zip and upload
    z = util_boto3.zip_and_upload(args_s3,
                                  ["result_precalc.json", "args_precalc.json"] + glob.glob(
                                      os.path.join(dirname, "*")), True, skip_upload=skip_upload)
    print(z)
    return z


def set_for_artificial(args_s3rnn, dirname):
    with zipfile.ZipFile(args_s3rnn) as z:
        z.extractall(dirname)
    # load wfa
    with open(os.path.join(dirname, "wfa.pickle"), "rb") as f:
        wfa: WFA.WFA = pickle.load(f)
    # load settings
    with open(os.path.join(dirname, "args.json"), "r") as f:
        j = json.load(f)
        args_embed_dim = j["embed_dim"]
        args_hidden_output_dims = j["hidden_output_dims"]
        args_batch_size = j["batch_size"]
        args_max_length = j["max_length"]
        print(
            f"embed_dim: {args_embed_dim}, hidden_output_dims: {args_hidden_output_dims}, batch_size: {args_batch_size}, max_length: {args_max_length}")
    # load words
    with open(os.path.join(dirname, "test.txt"), 'r') as f:
        words = [w.strip() for w in f]
    # eval
    regr = RNN.RNNRegression(len(wfa.alphabet),
                             args_embed_dim,
                             args_hidden_output_dims,
                             args_max_length)
    alphabet2id = Dataset.make_alphabet2id_dict(wfa.alphabet)
    return alphabet2id, args_max_length, regr, words

def set_for_paren(args_s3rnn, dirname):
    # alph = "()0123456789"

    with zipfile.ZipFile(args_s3rnn) as z:
        z.extractall(dirname)
    # load settings
    with open(os.path.join(dirname, "alphabet.txt"), "r") as f:
        alph = f.readline().strip()
        print("alphabet: ", alph)
    with open(os.path.join(dirname, "args.json"), "r") as f:
        j = json.load(f)
        args_embed_dim = j["embed_dim"]
        args_hidden_output_dims = j["hidden_output_dims"]
        args_batch_size = j["batch_size"]
        args_max_length = j["max_length"]
        print(
            f"embed_dim: {args_embed_dim}, hidden_output_dims: {args_hidden_output_dims}, batch_size: {args_batch_size}, max_length: {args_max_length}")
    # load words
    def cover_empty(s):
        if set(s) - set(alph):
            return ""
        return s
    with open(os.path.join(dirname, "test.txt"), 'r') as f:
        words = [cover_empty(w.strip().split()[0]) for w in f]
    # eval
    regr = RNN.RNNRegression(len(alph),
                             args_embed_dim,
                             args_hidden_output_dims,
                             args_max_length)
    alphabet2id = Dataset.make_alphabet2id_dict(alph)
    return alphabet2id, args_max_length, regr, words


def set_for_masaki(args_s3rnn, dirname):
    with zipfile.ZipFile(args_s3rnn) as z:
        shutil.rmtree(dirname)
        z.extractall(".")
        extdel = os.path.splitext(args_s3rnn)[0]
        shutil.move(extdel, dirname)
    with open(os.path.join(dirname, "alphabet.tsv"), "r") as f:
        alphabet = "".join(f.readline().split())
    # load settings
    args_embed_dim = 50
    args_hidden_output_dims = [50, 50]
    args_max_length = 40
    # load words
    with open(os.path.join(dirname, "test.txt"), 'r') as f:
        words = [w.strip() for w in f]
    # words = words[:10]  # temp
    # eval
    regr = RNN.RNNRegression(len(alphabet),
                             args_embed_dim,
                             args_hidden_output_dims,
                             args_max_length)
    alphabet2id = Dataset.make_alphabet2id_dict(alphabet)
    return alphabet2id, args_max_length, regr, words


def main():
    parser = argparse.ArgumentParser(
        description='Calc and count')

    parser.add_argument('--s3', action="store_true",
                        help='filename of s3')
    parser.add_argument('--paren', action="store_true",
                        help='needed if the rnn is made for paren experiment')
    parser.add_argument('rnn', default="",
                        help='filename of rnn.  If it is not found, downloaded from S3.')
    parser.add_argument('-o', default="result_precalc_rnn",
                        help='filename of output')
    args = parser.parse_args()

    global skip_upload
    skip_upload = not args.s3

    run(args_s3rnn=args.rnn,
        args_s3=args.o,
        paren=args.paren)


if __name__ == '__main__':
    main()
