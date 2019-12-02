import logging
import argparse
import util_boto3
import zipfile
import os
import json
import tensorflow as tf
import pickle
import RNN
import WFA
import Dataset

logger = logging.getLogger("talk_with_rnn")
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
handler_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)


def run(rnn, word, s3rnn, max_length):
    if s3rnn != "":
        dirname = "temp_talk_with_rnn"
        util_boto3.download(s3rnn)
        with zipfile.ZipFile(s3rnn) as z:
            z.extractall(dirname)
        if s3rnn.startswith("hualalai_masaki"):
            dirname = os.path.join(dirname, os.path.splitext(s3rnn)[0])
    else:
        dirname = rnn

    # load wfa
    path_wfa = os.path.join(dirname, "wfa.pickle")
    path_tsv = os.path.join(dirname, "alphabet.tsv")
    if os.path.exists(path_wfa):
        with open(path_wfa, "rb") as f:
            wfa: WFA.WFA = pickle.load(f)
        alphabet = wfa.alphabet
    elif os.path.exists(path_tsv):
        with open(path_tsv, "r") as f:
            x = f.readline()
            alphabet = x.replace(" ", "")
        with open(os.path.join(dirname, "args.json"), "w") as f:
            json.dump({"embed_dim": 50, "hidden_output_dims": [50, 50]}, f)
    else:
        assert False

    # load rnn setting
    with open(os.path.join(dirname, "args.json"), "r") as f:
        d = json.load(f)
        embed_dim = d["embed_dim"]
        hidden_output_dims = d["hidden_output_dims"]

    alphabet2id = Dataset.make_alphabet2id_dict(alphabet)
    regr = RNN.RNNRegression(len(alphabet),
                             embed_dim,
                             hidden_output_dims,
                             max_length)

    csm = RNN.ContinuousStateMachine(regr, None, alphabet2id)
    saver = tf.train.Saver(max_to_keep=0)

    with tf.Session() as sess:
        if os.path.isdir(dirname):
            checkpoint = tf.train.latest_checkpoint(dirname)
        else:
            assert False

        csm._sess = sess
        saver.restore(sess, checkpoint)
        print(csm.get_value(word))


def main():
    parser = argparse.ArgumentParser(
        description='Extract WFA from RNN')
    parser.add_argument('RNN',
                        help=('The checkpoint directory where trained models '
                              'are stored, or the checkpoint file storing the '
                              'model to be evalauted.  The configuration is loaded from args.json'))
    parser.add_argument('word',
                        help=('word'))
    parser.add_argument('--s3rnn', default="",
                        help="RNN in S3.  If it is specified, RNN option is ignored.")
    parser.add_argument('--max_length', default=50, type=int,
                        help="max_length in RNN")
    args = parser.parse_args()

    run(args.RNN, args.word, args.s3rnn, args.max_length)


if __name__ == '__main__':
    logger.info("Starting")
    main()
