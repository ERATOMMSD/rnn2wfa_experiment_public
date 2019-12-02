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

logger = logging.getLogger("talk_with_wfa")
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
handler_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)


def run(wfa, word, s3wfa):
    if s3wfa != "":
        dirname = "temp_talk_with_wfa"
        util_boto3.download(s3wfa)
        with zipfile.ZipFile(s3wfa) as z:
            z.extractall(dirname)
        if s3wfa.startswith("hualalai_masaki"):
            dirname = os.path.join(dirname, os.path.splitext(s3wfa)[0])
        filename = os.path.join(dirname, wfa)
    else:
        filename = wfa
        dirname = os.path.dirname(filename)


    # load wfa
    path_wfa = filename
    path_tsv = os.path.join(dirname, "alphabet.tsv")
    print(path_wfa)
    if os.path.exists(path_wfa):
        with open(path_wfa, "rb") as f:
            wfa: WFA.WFA = pickle.load(f)
            wfa.callings = set()
        alphabet = wfa.alphabet
    elif os.path.exists(path_tsv):
        with open(path_tsv, "r") as f:
            x = f.readline()
            alphabet = x.replace(" ", "")
        with open(os.path.join(dirname, "args.json"), "w") as f:
            json.dump({"embed_dim": 50, "hidden_output_dims": [50, 50]}, f)
    else:
        assert False

    print(wfa.get_value(word))


def main():
    parser = argparse.ArgumentParser(
        description='WFA')
    parser.add_argument('WFA',
                        help=('The checkpoint directory where trained models '
                              'are stored, or the checkpoint file storing the '
                              'model to be evalauted.  The configuration is loaded from args.json'))
    parser.add_argument('word',
                        help=('word'))
    parser.add_argument('--s3wfa', default="",
                        help="wfa in S3.  If it is specified, RNN option is ignored.")

    args = parser.parse_args()

    run(args.WFA, args.word, args.s3wfa)


if __name__ == '__main__':
    logger.info("Starting")
    main()
