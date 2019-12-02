import logging
import argparse
import Lstar
import util_boto3
import zipfile
import os
import shutil
import QuantitativeObservationTable
import json
import equiv_query
import equiv_query_regr
import equiv_query_random
import equiv_query_search
import tensorflow as tf
import pickle
import RNN
import WFA
import Dataset
import glob
from typing import *
import RNN_ADFA

logger = logging.getLogger("rnn2wfa")
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
handler_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(handler_format)

stream_handler_file = logging.FileHandler(filename="rnn2wfa.log", mode="w")
stream_handler_file.setLevel(logging.DEBUG)
stream_handler_file.setFormatter(handler_format)

logger.addHandler(stream_handler)
logger.addHandler(stream_handler_file)

default_tol_rank_init = 0.1
default_tol_rank_decay_rate = 0.5
default_tol_rank_lower_bound = 1e-7
default_qot_timeout = None
default_timeout = 1000
default_max_length = 100

skip_upload = False


def run(args_s3rnn, args_RNN, args_tol_rank_init, args_tol_rank_decay_rate, args_tol_rank_lower_bound, args_qot_timeout,
        args_eqq_type, args_eqq_param, args_save_process, args_timeout, args_max_length, args_s3):
    tf.reset_default_graph()
    dirname = "temp_rnn2wfa"
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)

    params = make_params(args_eqq_param, args_eqq_type, args_qot_timeout, args_save_process, args_timeout,
                         args_tol_rank_decay_rate, args_tol_rank_init, args_tol_rank_lower_bound, dirname)

    if args_RNN == "artificial":
        args_RNN = dirname
        result = load_artificial_rnn_and_run(args_RNN, args_max_length, args_s3rnn, dirname, params)
    elif args_RNN.startswith("adfa"):
        splitted = args_RNN.split("_")
        proj = splitted[1]
        top_n = int(splitted[2])
        rnn = RNN_ADFA.RNNAdfa(proj, top_n)
        result = Lstar.run_quantitative_lstar(rnn, params)
    else:
        assert False

    # save extracted
    with open(os.path.join(dirname, "wfa_extracted.pickle"), "wb") as f:
        pickle.dump(result.wfa, f)
    with open(os.path.join(dirname, "args_rnn2wfa.json"), "w") as f:
        # args_s3rnn, args_RNN, args_tol_rank_init, args_tol_rank_decay_rate, args_tol_rank_lower_bound, args_qot_timeout,
        #         args_eqq_type, args_eqq_param, args_save_process, args_timeout, args_max_length, args_s3
        d = {"s3rnn": args_s3rnn,
             "RNN": args_RNN,
             "tol_rank_init": args_tol_rank_init,
             "tol_rank_decay_rate": args_tol_rank_decay_rate,
             "tol_rank_lower_bound": args_tol_rank_lower_bound,
             "qot_timeout": args_qot_timeout,
             "eqq_type": args_eqq_type,
             "eqq_param": args_eqq_param,
             "save_process": args_save_process,
             "timeout": args_timeout,
             "max_length": args_max_length,
             "s3": args_s3}
        json.dump(d, f)
    with open(os.path.join(dirname, "statistics.json"), "w") as f:
        json.dump(result.stat.to_dict(), f)
    shutil.copy(args_eqq_param, dirname)

    # zip and upload (later statistics can be added)
    z = util_boto3.zip_and_upload(args_s3,
                                  glob.glob(
                                      os.path.join(dirname, "*")) + ["rnn2wfa.log"], True,
                                  skip_upload=skip_upload)
    print(z)
    return z


def load_artificial_rnn_and_run(args_RNN, args_max_length, args_s3rnn, dirname, params):
    # load rnn
    util_boto3.download(args_s3rnn)
    with zipfile.ZipFile(args_s3rnn) as z:
        if "hualalai_masaki" in args_s3rnn:
            z.extractall(".")
            shutil.rmtree(dirname)
            shutil.move(os.path.splitext(args_s3rnn)[0], dirname)
            print(os.path.splitext(args_s3rnn)[0])
        else:
            z.extractall(dirname)
    # if args_s3rnn.startswith("hualalai_masaki"):
    #     dirname = os.path.join(dirname, os.path.splitext(args_s3rnn)[0])
    # load wfa
    path_wfa = os.path.join(dirname, "wfa.pickle")
    path_tsv = os.path.join(dirname, "alphabet.tsv")
    path_txt = os.path.join(dirname, "alphabet.txt")
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
    elif os.path.exists(path_txt):
        with open(path_txt, "r") as f:
            alphabet = f.readline().strip()
            print("alphabet:", alphabet)
    elif "paren" in args_s3rnn:
        alphabet = "()0123456789" # see make_paren_data
    else:
        assert False
    # load rnn setting
    with open(os.path.join(dirname, "args.json"), "r") as f:
        d = json.load(f)
        embed_dim = d["embed_dim"]
        hidden_output_dims = d["hidden_output_dims"]
        # max_length = d["max_length"]
    alphabet2id = Dataset.make_alphabet2id_dict(alphabet)
    regr = RNN.RNNRegression(len(alphabet),
                             embed_dim,
                             hidden_output_dims,
                             args_max_length)
    csm = RNN.ContinuousStateMachine(regr, None, alphabet2id)
    saver = tf.train.Saver(max_to_keep=0)
    # config = tf.ConfigProto(
    #     gpu_options=tf.GPUOptions(
    #         per_process_gpu_memory_fraction=0.5
    #     )
    # )
    # with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        if os.path.isdir(args_RNN):
            checkpoint = tf.train.latest_checkpoint(args_RNN)
        else:
            assert False

        csm._sess = sess
        saver.restore(sess, checkpoint)
        result = Lstar.run_quantitative_lstar(csm, params)
    return result


def make_params(args_eqq_param, args_eqq_type, args_qot_timeout, args_save_process, args_timeout,
                args_tol_rank_decay_rate, args_tol_rank_init, args_tol_rank_lower_bound, dirname):
    # make params
    params_qot = QuantitativeObservationTable.QuantitativeObservationTableParameters(False, args_tol_rank_init,
                                                                                     args_tol_rank_decay_rate,
                                                                                     args_tol_rank_lower_bound,
                                                                                     args_qot_timeout)
    name2eqqa_eqqp: Dict[str, Tuple[Type[equiv_query.EquivalenceQueryAnswererBase], Any]] = {
        "regr": (equiv_query_regr.EquivalenceQueryAnswerer, equiv_query_regr.EquivalenceQueryParameters),
        # "learnaut": (equiv_query_regr.EquivalenceQueryAnswererLearnAut, equiv_query_regr.EquivalenceQueryParameters),
        "random": (equiv_query_random.EquivalenceQueryAnswerer, equiv_query_random.EquivalenceQueryParameters),
        "search": (equiv_query_search.EquivalenceQueryAnswerer, equiv_query_search.EquivalenceQueryParameters)
    }
    eqqa_class, eqqp_class = name2eqqa_eqqp[args_eqq_type]
    with open(args_eqq_param, "r") as f:
        opt_in_json: Dict[str, Any] = json.load(f)
        logger.info("Given option is:" + str(opt_in_json))
        eqqp = eqqp_class(**opt_in_json)

    def eqqa_maker(rnn):
        return eqqa_class(rnn, eqqp, dirname)

    save_process_to = dirname if args_save_process else None
    params = Lstar.LstarParameters(params_qot, eqqa_maker, None if args_timeout < 0 else args_timeout,
                                   save_process_to)
    return params


def main():
    parser = argparse.ArgumentParser(
        description='Extract WFA from RNN')
    # parser.add_argument('RNN',
    #                     help=('The checkpoint directory where trained models '
    #                           'are stored, or the checkpoint file storing the '
    #                           'model to be evalauted'))
    parser.add_argument('eqq_type',
                        help=('Name of EquivalenceQuery'),
                        choices = ["regr", "sample", "search"]) # regr, sample, search
    parser.add_argument('eqq_param',
                        help=('parameter dictionary of EquivalenceQuery in json string'))
    parser.add_argument('--tol_rank_init', default=default_tol_rank_init, type=float,
                        help='tol_rank_init in QuantitativeObservationTable')
    parser.add_argument('--tol_rank_decay_rate', default=default_tol_rank_decay_rate, type=float,
                        help='tol_rank_decay_rate in QuantitativeObservationTable')
    parser.add_argument('--tol_rank_lower_bound', default=default_tol_rank_lower_bound, type=float,
                        help='tol_rank_lower_bound in QuantitativeObservationTable')
    parser.add_argument('--qot_timeout', default=default_qot_timeout, type=int,
                        help='timeout in QuantitativeObservationTable')
    parser.add_argument('--timeout', default=default_timeout, type=int,
                        help='Timeout.  If it is negative, the timeout is disabled.')
    parser.add_argument('--s3', action="store_true",
                        help="upload to s3")
    parser.add_argument('-o', default="result_rnn2wfa.zip",
                        help="filename of output")
    parser.add_argument('rnn',
                        help="RNN.  If it is not found, downloaded from s3.")
    parser.add_argument('--max_length', type=int, default=default_max_length,
                        help="max length that RNN can take")
    parser.add_argument('--save_process', type=bool, default=False,
                        help="directory to save the process WFAs in.  If it is empty, they are not saved.")
    args = parser.parse_args()


    global skip_upload
    skip_upload = not args.s3


    run(args.rnn, "artificial", args.tol_rank_init, args.tol_rank_decay_rate, args.tol_rank_lower_bound,
        args.qot_timeout, args.eqq_type, args.eqq_param, args.save_process, args.timeout, args.max_length, args.o)


if __name__ == '__main__':
    logger.info("Starting")
    main()
