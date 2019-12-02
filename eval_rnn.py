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
import csv
import json
import glob
import time
import util

default_embeded_dim = 50
default_hidden_output_dims = [50]
default_batch_size = 128
default_max_length = 50
default_eps = 0.05
default_eps_relative = False
default_skip_train = False


def eps_equality_absolute(x: float, y: float, eps: float) -> bool:
    return abs(x - y) < eps


def eps_equality_relative(x: float, y: float, eps: float) -> bool:
    ratio = 1.0 + eps
    less = min(x / ratio, x * ratio)
    greater = max(x / ratio, x * ratio)
    return less <= y <= greater


class EvaluateResult(NamedTuple):
    str2wfavalue: List[float]
    str2rnnvalue: List[float]
    loss: float
    ave_loss: float
    n_equals: int
    time_wfa: float
    time_rnn: float

    def get_dict(self) -> Dict[str, Any]:
        return {"loss": float(self.loss), "ave_loss": float(self.ave_loss), "n_equals": int(self.n_equals),
                "time_wfa": float(self.time_wfa),
                "time_rnn": float(self.time_rnn)}


def evaluate(regression: RNN.RNNRegression,
             max_length: int,
             batch_size: int,
             eval_data: List[Tuple[np.ndarray, float]],
             checkpoint: str,
             equality: Callable[[Tuple[float, float]], bool]) -> EvaluateResult:
    input_op = tf.placeholder(tf.float32, shape=(None, max_length))
    label_op = tf.placeholder(tf.float32, shape=(None))
    loss_op, output_op = regression.loss(input_op, label_op)
    saver = tf.train.Saver(max_to_keep=0)

    label_log = []
    output_log = []

    with tf.Session() as sess:
        if os.path.isdir(checkpoint):
            checkpoint = tf.train.latest_checkpoint(checkpoint)
        saver.restore(sess, checkpoint)

        n_batches = math.ceil(len(eval_data) / batch_size)
        loss = 0.
        n_equals = 0
        start = time.time()
        for i in progressbar.progressbar(range(n_batches)):
            batch = eval_data[i * batch_size:(i + 1) * batch_size]
            input_data, label_data = list(zip(*batch))  # unzip
            batch_loss, output_data = sess.run([loss_op, output_op],
                                               feed_dict={input_op: input_data,
                                                          label_op: label_data})
            loss += batch_loss
            output_data = list(output_data)
            assert (len(output_data) == len(label_data))
            n_equals += len(list(filter(equality, zip(label_data, output_data))))
            label_log += label_data
            output_log += output_data
        time_rnn = time.time() - start
        ave_loss = loss / (len(eval_data) / batch_size)
        print(f'Loss (total): {loss}')
        print(f'Loss (average): {ave_loss}')
        print(f'# of equals / # of data: {n_equals} / {len(eval_data)}')
    return EvaluateResult(label_log, output_log, loss, ave_loss, n_equals, 0, time_rnn)


def evaluate_and_make_table(regression: RNN.RNNRegression,
                            max_length: int,
                            batch_size: int,
                            words_file: str,
                            wfa: WFA.WFA,
                            alphabet2id: Dict[str, int],
                            checkpoint: str,
                            equality: Callable[[Tuple[float, float]], bool]) -> Tuple[
    Iterable[Tuple[str, float, float]], EvaluateResult]:
    start = time.time()
    eval_data = Dataset.load_data(words_file,
                                  wfa,
                                  alphabet2id,
                                  max_length)
    time_wfa = time.time() - start
    with open(words_file, 'r') as f:
        words = [w.strip() for w in f]
    res_eval = evaluate(regression, max_length, batch_size, eval_data, checkpoint, equality)
    res_eval = EvaluateResult(res_eval.str2wfavalue, res_eval.str2rnnvalue, res_eval.loss, res_eval.ave_loss,
                              res_eval.n_equals, time_wfa, res_eval.time_rnn)

    table = zip(words, res_eval.str2wfavalue, [float(x) for x in res_eval.str2rnnvalue])
    return table, res_eval


def find_optimizer(opt: str) -> tf.train.Optimizer:
    opt_dict = {
        'adam': tf.train.AdamOptimizer(),
    }
    return opt_dict[opt]


def run(args_s3rnn, args_s3wfa, args_embed_dim, args_hidden_output_dims, args_batch_size, args_max_length,
        args_eps_relative, args_eps, args_s3, args_skip_train):
    dirname = "temp_eval_rnn"
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)

    # download
    if args_s3wfa == "":
        util_boto3.download(args_s3rnn)
        with zipfile.ZipFile(args_s3rnn) as z:
            z.extractall("temp_eval_rnn")
    else:
        util_boto3.download(args_s3wfa)
        with zipfile.ZipFile(args_s3wfa) as z:
            z.extractall("temp_eval_rnn")

    # load words
    with open("temp_eval_rnn/args.json", "r") as f:
        j = json.load(f)
        if args_embed_dim == 0:
            args_embed_dim = j["embed_dim"]
        if args_hidden_output_dims == [0] or args_hidden_output_dims == 0:
            args_hidden_output_dims = j["hidden_output_dims"]
        if args_batch_size == 0:
            args_batch_size = j["batch_size"]
        if args_max_length == 0:
            args_max_length = j["max_length"]

    # load wfa
    if args_s3wfa == "":
        with open("temp_eval_rnn/wfa.pickle", "rb") as f:
            wfa: WFA.WFA = pickle.load(f)
    else:
        util_boto3.download(args_s3wfa)
        with open("temp_eval_rnn/wfa_extracted.pickle", "rb") as f:
            wfa: WFA.WFA = pickle.load(f)
    wfa.callings = set()

    # eval
    regr = RNN.RNNRegression(len(wfa.alphabet),
                             args_embed_dim,
                             args_hidden_output_dims,
                             args_max_length)
    equality_fun = eps_equality_relative if args_eps_relative \
        else eps_equality_absolute
    equality = lambda x: equality_fun(x[0], x[1], args_eps)
    alphabet2id = Dataset.make_alphabet2id_dict(wfa.alphabet)

    # eval
    if not args_skip_train:
        print("-" * 30 + "TRAIN" + "-" * 30)
        res_train = evaluate_and_make_table(regr, args_max_length, args_batch_size, "temp_eval_rnn/train.txt", wfa,
                                            alphabet2id, dirname, equality)
    else:
        res_train = [[], EvaluateResult([], [], 0, 0, 0, 0, 0)]
    print("-" * 30 + "TEST" + "-" * 30)
    res_test = evaluate_and_make_table(regr, args_max_length, args_batch_size, "temp_eval_rnn/test.txt", wfa,
                                       alphabet2id, dirname, equality)

    # write
    # write train table
    with open("data_train.csv", "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(res_train[0])
    # write test table
    with open("data_test.csv", "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(res_test[0])
    # write train result
    with open("result_eval_rnn.json", "w") as f:
        json.dump({"train": res_train[1].get_dict(), "test": res_test[1].get_dict()}, f)
    util.notify_slack(str(res_test[1].get_dict()))
    # write args
    with open("args_eval.json", "w") as f:
        d = {"s3rnn": args_s3rnn, "s3wfa": args_s3wfa, "embed_dim": args_embed_dim,
             "hidden_output_dims": args_hidden_output_dims, "batch_size": args_batch_size,
             "max_length": args_max_length, "eps_relative": args_eps_relative, "eps": args_eps, "s3": args_s3}
        json.dump(d, f)

    # zip and upload
    z = util_boto3.zip_and_upload(args_s3,
                                  ["data_train.csv", "data_test.csv", "result_eval_rnn.json",
                                   "args_eval.json"] + glob.glob(
                                      os.path.join(dirname, "*")), True)
    print(z)
    return z


def main():
    parser = argparse.ArgumentParser(
        description='Train a RNN on a WFA')
    parser.add_argument('RNN',
                        help=('The checkpoint directory where trained models '
                              'are stored, or the checkpoint file storing the '
                              'model to be evalauted'))
    parser.add_argument('WFA',
                        help=('The pickle file serializing a WFA '
                              'on which the RNN was trained'))
    parser.add_argument('eval',
                        help=('The file for evaluation data '
                              '(one per line)'))

    parser.add_argument('--embed_dim', default=default_embeded_dim, type=int,
                        help=('The dimension of the output of the embedding of '
                              'the RNN'))
    parser.add_argument('--hidden_output_dims', default=default_hidden_output_dims,
                        type=lambda x: [int(n) for n in x.split(',')],
                        help=('The dimensions of the outputs of '
                              'the hidden layers of the RNN, '
                              "separated by ',' "
                              '(the empty means no layers)'))
    parser.add_argument('--batch_size', type=int, default=default_batch_size,
                        help='The size of a batch')
    parser.add_argument('--max_length', type=int, default=default_max_length,
                        help=('The maximum length of a word'))
    parser.add_argument('--eps', type=float, default=default_eps,
                        help='The allowable error')
    parser.add_argument('--eps_relative', action='store_true',
                        help='Whehter the eps error is relative or absolute')
    parser.add_argument('--s3', default="",
                        help='filename of s3')
    parser.add_argument('--s3rnn', default="",
                        help='filename of rnn in S3')
    parser.add_argument('--s3wfa', default="",
                        help='filename of wfa in S3.  If omitted, the wfa used for WFA2RNN is used.')
    parser.add_argument('--skip_train', type=bool, default=default_skip_train,
                        help='filename of wfa in S3.  If omitted, the wfa used for WFA2RNN is used.')
    args = parser.parse_args()

    if args.s3 == "":
        if not os.path.exists(args.RNN):
            print(f'"{args.RNN}" does not exist')
            return

        with open(args.WFA, 'rb') as f:
            wfa = pickle.load(f)

        alphabet2id = Dataset.make_alphabet2id_dict(wfa.alphabet)
        eval_data = Dataset.load_data(args.eval,
                                      wfa,
                                      alphabet2id,
                                      args.max_length)
        regr = RNN.RNNRegression(len(wfa.alphabet),
                                 args.embed_dim,
                                 args.hidden_output_dims,
                                 args.max_length)
        equality_fun = eps_equality_relative if args.eps_relative \
            else eps_equality_absolute
        equality = lambda x: equality_fun(x[0], x[1], args.eps)
        evaluate(regr,
                 args.max_length,
                 args.batch_size,
                 eval_data,
                 args.RNN,
                 equality)
    else:
        run(args_s3rnn=args.s3rnn,
            args_s3wfa=args.s3wfa,
            args_embed_dim=args.embed_dim,
            args_hidden_output_dims=args.hidden_output_dims,
            args_batch_size=args.batch_size,
            args_max_length=args.max_length,
            args_eps_relative=args.eps_relative,
            args_eps=args.eps,
            args_s3=args.s3,
            args_skip_train=args.skip_train)


if __name__ == '__main__':
    main()
