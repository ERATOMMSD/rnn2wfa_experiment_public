import argparse
import glob
import json
import os
import pickle
import shutil
import zipfile
from typing import *

import Dataset
import RNN
import WFA
import util
import util_boto3
from train_rnn_on_wfa import train
import eval_rnn


skip_upload = False

def split_line(x):
    splitted = x.split()
    # print(splitted)
    if len(splitted) == 1:
        return "", float(splitted[0])
    elif len(splitted) == 2:
        return splitted[0], float(splitted[1])
    else:
        assert False


def run_s3(s3: str, s3words: str, max_length: int, embed_dim: int,
           hidden_output_dims: Union[int, Iterable[int]], optimizer: str, n_epochs: int, batch_size: int,
           save_interval: int) -> str:
    dirname = "temp_train_rnn_on_file"
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)

    # load words
    util_boto3.download(s3words)
    with zipfile.ZipFile(s3words) as z:
        z.extractall("words")
    with open("words/train.txt", "r") as f:
        words_train: List[Tuple[str, float]] = [split_line(x) for x in f.readlines()]
        print("Training: ")
        for i in words_train[:5]:
            print(f"{i[0]} : {i[1]}")
        for i in words_train:
            print(i)
            assert len(i[0]) <= max_length
    with open("words/test.txt", "r") as f:
        words_test: List[Tuple[str, float]] = [split_line(x) for x in f.readlines()]
        print("Testing: ")
        for i in words_test[:5]:
            print(f"{i[0]} : {i[1]}")
        for i in words_test:
            print(i)
            assert len(i[0]) <= max_length

    with open("words/alphabet.txt", "r") as f:
        alphabet = f.readline().strip()
        print(f"Alphabet is {alphabet}")

    # train
    alphabet2id = Dataset.make_alphabet2id_dict(alphabet)
    regr = RNN.RNNRegression(len(alphabet),
                             embed_dim,
                             list(hidden_output_dims),
                             max_length)

    def words_float_to_nparray_float(words):
        return [(Dataset.word2vec(x[0], alphabet2id, max_length), x[1]) for x in words]

    train(regr,
          max_length,
          optimizer,
          n_epochs,
          batch_size,
          words_float_to_nparray_float(words_train),
          save_interval,
          dirname)

    # calc results
    # try:
    ev = eval_rnn.evaluate(regr, max_length, batch_size, words_float_to_nparray_float(words_test), dirname,
                           lambda p: eval_rnn.eps_equality_absolute(p[0], p[1], 0.05))
    # except:
    #     ev = None
    #     print("Error happend in evaluation!")

    # args
    a = {"s3": s3,
         "s3words": s3words, "max_length": max_length, "embed_dim": embed_dim,
         "hidden_output_dims": hidden_output_dims, "optimizer": optimizer, "n_epochs": n_epochs,
         "batch_size": batch_size,
         "save_interval": save_interval}
    with open("args.json", "w") as f:
        json.dump(a, f)
    with open("eval_rnn.json", "w") as f:
        json.dump(ev.get_dict(), f)

    # file
    files = glob.glob(os.path.join(dirname, "*")) + ["words/train.txt", "words/test.txt",
                                                     "args.json", "eval_rnn.json", "words/alphabet.txt"]

    z = util_boto3.zip_and_upload(s3, files, True, skip_upload=skip_upload)
    mes = f"Saved to {z}.  The option were {str(a)}"
    print(mes)
    util.notify_slack(mes)
    print(z)
    return z


def main():
    parser = argparse.ArgumentParser(
        description='Train a RNN on a WFA')
    parser.add_argument('traindata',
                        help='The data file to learn from.  It consist of lines of pairs of a word and the value.  If it is not found, downloaded from s3.')
    parser.add_argument('--embed_dim', default=50, type=int,
                        help='The dimension of the output of the embedding')
    parser.add_argument('--hidden_output_dims', default="50,50",
                        type=lambda x: [int(n) for n in x.split(',')],
                        help=('The dimensions of the outputs of '
                              'the hidden layers of the RNN, '
                              "separated by ',' "
                              '(the empty means no layers)'))
    parser.add_argument('--batch_size', type=int, default=1,
                        help='The size of a batch')
    parser.add_argument('--n_epochs', type=int, default=5,
                        help='The number of epochs')
    parser.add_argument('--optimizer', default='adam',
                        choices=['adam'],
                        help='The optimizer')
    parser.add_argument('--max_length', type=int, default=20,
                        help='The maximum length of a word')
    parser.add_argument('--save_interval', type=int, default=1,
                        help=('Save trained models '
                              'every SAVE_INTERVAL training epochs'))
    parser.add_argument('--s3', action="store_true",
                        help="upload to s3")
    parser.add_argument('-o', type=str, default="result_train_rnn_on_file.zip",
                        help="filename of output")
    #    parser.add_argument('--s3words', default="",
    #                        help="words set in S3")
    #    parser.add_argument('--s3wfa', default="",
    #                        help="WFA in S3")
    args = parser.parse_args()

    global skip_upload
    skip_upload = not args.s3

    run_s3(args.o, args.traindata, args.max_length, args.embed_dim, args.hidden_output_dims,
           args.optimizer, args.n_epochs, args.batch_size, args.save_interval)


if __name__ == '__main__':
    main()
