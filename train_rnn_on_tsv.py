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


def run_s3(s3: str, s3words: str, s3wfa: str, max_length: int, embed_dim: int,
           hidden_output_dims: Union[int, Iterable[int]], optimizer: str, n_epochs: int, batch_size: int,
           save_interval: int) -> str:
    dirname = "temp_train_rnn_on_wfa"
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)

    # load words
    util_boto3.download(s3words)
    with zipfile.ZipFile(s3words) as z:
        z.extractall("words")
    with open("words/train.txt", "r") as f:
        words_train: List[str] = [x.strip() for x in f.readlines()]
    with open("words/test.txt", "r") as f:
        words_test: List[str] = [x.strip() for x in f.readlines()]

    # load wfa
    util_boto3.download(s3wfa)
    with zipfile.ZipFile(s3wfa) as z:
        z.extractall("wfa")
    with open("wfa/wfa.pickle", "rb") as f:
        wfa: WFA.WFA = pickle.load(f)

    # train
    alphabet2id = Dataset.make_alphabet2id_dict(wfa.alphabet)
    train_data = Dataset.load_data("words/train.txt",
                                   wfa,
                                   alphabet2id,
                                   max_length)
    regr = RNN.RNNRegression(len(wfa.alphabet),
                             embed_dim,
                             hidden_output_dims,
                             max_length)
    train(regr,
          max_length,
          optimizer,
          n_epochs,
          batch_size,
          train_data,
          save_interval,
          dirname)

    # calc results

    # args
    a = {"s3": s3,
         "s3words": s3words, "s3wfa": s3wfa, "max_length": max_length, "embed_dim": embed_dim,
         "hidden_output_dims": hidden_output_dims, "optimizer": optimizer, "n_epochs": n_epochs,
         "batch_size": batch_size,
         "save_interval": save_interval}
    with open("args.json", "w") as f:
        json.dump(a, f)

    # file
    files = glob.glob(os.path.join(dirname, "*")) + ["words/train.txt", "words/test.txt", "wfa/wfa.pickle",
                                                     "args.json"]

    z = util_boto3.zip_and_upload(s3, files, True)
    mes = f"Saved to {z}.  The option were {str(a)}"
    print(mes)
    util.notify_slack(mes)
    print(z)
    return z


def main():
    parser = argparse.ArgumentParser(
        description='Train a RNN on a WFA')
    parser.add_argument('alphabet',
                        help='The tsv file for the alphabet')
    parser.add_argument('train',
                        help=('The tsv file for the training data'
                              '(one per line)'))
    parser.add_argument('checkpoint_dir',
                        help='The directory where trained models are stored')
    parser.add_argument('--embed_dim', default=50, type=int,
                        help='The dimension of the output of the embedding')
    parser.add_argument('--hidden_output_dims', default=[50],
                        type=lambda x: [int(n) for n in x.split(',')],
                        help=('The dimensions of the outputs of '
                              'the hidden layers of the RNN, '
                              "separated by ',' "
                              '(the empty means no layers)'))
    parser.add_argument('--batch_size', type=int, default=32,
                        help='The size of a batch')
    parser.add_argument('--n_epochs', type=int, default=5,
                        help='The number of epochs')
    parser.add_argument('--optimizer', default='adam',
                        choices=['adam'],
                        help='The optimizer')
    parser.add_argument('--max_length', type=int, default=50,
                        help='The maximum length of a word')
    parser.add_argument('--save_interval', type=int, default=1,
                        help=('Save trained models '
                              'every SAVE_INTERVAL training epochs'))
    parser.add_argument('--s3', default="",
                        help="zipfile of the trained RNN")
    #    parser.add_argument('--s3words', default="",
    #                        help="words set in S3")
    #    parser.add_argument('--s3wfa', default="",
    #                        help="WFA in S3")
    args = parser.parse_args()

    if args.s3 == "":
        if not os.path.isdir(args.checkpoint_dir):
            if os.path.exists(args.checkpoint_dir):
                print(f'"{args.checkpoint_dir} is not a directory')
                return
            else:
                os.makedirs(args.checkpoint_dir)

        alphabet = Dataset.parse_alphabet_tsv(args.alphabet)

        alphabet2id = Dataset.make_alphabet2id_dict(alphabet)
        train_data = Dataset.load_data_tsv(args.train,
                                           alphabet2id,
                                           args.max_length)
        regression = RNN.RNNRegression(len(alphabet),
                                       args.embed_dim,
                                       args.hidden_output_dims,
                                       args.max_length)
        train(regression,
              args.max_length,
              args.optimizer,
              args.n_epochs,
              args.batch_size,
              train_data,
              args.save_interval,
              args.checkpoint_dir)
    else:
        print('S3 is not implemented yet!!')
        exit(1)
        run_s3(args.s3, args.s3words, args.s3wfa, args.max_length, args.embed_dim, args.hidden_output_dims,
               args.optimizer, args.n_epochs, args.batch_size, args.save_interval)


if __name__ == '__main__':
    main()
