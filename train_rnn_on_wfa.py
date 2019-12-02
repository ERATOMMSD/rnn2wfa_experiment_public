import argparse
import math
import os
import pickle
import random
import re
from typing import *
import numpy as np
import progressbar
import tensorflow as tf
import WFA
import RNN
import Dataset
import util_boto3
import zipfile
import glob
import shutil
import json
import util

skip_upload = False


def train(regression: RNN.RNNRegression,
          max_length: int,
          opt: str,
          n_epochs: int,
          batch_size: int,
          train_data: List[Tuple[np.ndarray, float]],
          save_interval: int,
          checkpoint_dir: str) -> Any:
    input_op = tf.placeholder(tf.float32, shape=(None, max_length))
    label_op = tf.placeholder(tf.float32, shape=(None))
    loss_op, _ = regression.loss(input_op, label_op)
    opt_op = find_optimizer(opt)
    train_op = opt_op.minimize(loss_op)
    saver = tf.train.Saver(max_to_keep=0)

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.5
        )
    )

    with tf.Session(config=config) as sess:
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if checkpoint is None:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            cur_epoch = 0
        else:
            prev_epoch = re.search(r'\d+', os.path.basename(checkpoint))[0]
            cur_epoch = int(prev_epoch) + 1
            saver.restore(sess, checkpoint)

        n_batches = math.ceil(len(train_data) / batch_size)
        for e in range(cur_epoch, cur_epoch + n_epochs):
            random.shuffle(train_data)
            epoch_loss = 0.
            for i in progressbar.progressbar(range(n_batches)):
                batch = train_data[i * batch_size:(i + 1) * batch_size]
                input_data, label_data = list(zip(*batch))  # unzip
                _, batch_loss = sess.run([train_op, loss_op],
                                         feed_dict={input_op: input_data,
                                                    label_op: label_data})
                epoch_loss += batch_loss

            train_loss = epoch_loss / (len(train_data) / batch_size)
            print('train_loss @ {} epoch = {}'.format(e, train_loss))
            if e % save_interval == 0:
                model_file = os.path.join(checkpoint_dir, 'model')
                saver.save(sess, model_file, global_step=e)


def find_optimizer(opt: str) -> tf.train.Optimizer:
    opt_dict = {
        'adam': tf.train.AdamOptimizer(),
    }
    return opt_dict[opt]


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
        wfa.callings = set()

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

    z = util_boto3.zip_and_upload(s3, files, True, skip_upload=skip_upload)
    mes = f"Saved to {z}.  The option were {str(a)}"
    print(mes)
    util.notify_slack(mes)
    print(z)
    return z


def main():
    parser = argparse.ArgumentParser(
        description='Train a RNN on a WFA')

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
                        help=('The number of epochs'))
    parser.add_argument('--optimizer', default='adam',
                        choices=['adam'],
                        help='The optimizer')
    parser.add_argument('--max_length', type=int, default=20,
                        help=('The maximum length of a word'))
    parser.add_argument('--save_interval', type=int, default=1,
                        help=('Save trained models '
                              'every SAVE_INTERVAL training epochs'))
    parser.add_argument('--s3', action="store_true",
                        help="upload to s3")
    parser.add_argument('-o', type=str, default="result_train_rnn_on_wfa.zip",
                        help="filename of output")
    parser.add_argument('--words', default="",
                        help="words set.  If it is not found, downloaded from S3")
    parser.add_argument('--wfa', default="",
                        help="WFA file.   If it is not found, downloaded from S3")
    args = parser.parse_args()

    global skip_upload
    skip_upload = not args.s3

    z = run_s3(args.o, args.words, args.wfa, args.max_length, args.embed_dim, args.hidden_output_dims,
           args.optimizer, args.n_epochs, args.batch_size, args.save_interval)
    print(z)


if __name__ == '__main__':
    main()
