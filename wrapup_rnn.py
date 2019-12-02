import csv
import zipfile
import util_boto3
import os.path
import pickle
import WFA
import json
import itertools
import numpy as np
import eval_rnn
import precalc_rnn
import Dataset
import tensorflow as tf

# rnns = """20190513093734_rnn_no5_abcdef_10_3.zip
# 20190513115614_rnn_no3_abcdef_10_3.zip
# 20190513094050_rnn_no5_abcd_10_3.zip
# 20190513120311_rnn_no2_abcdef_10_3.zip
# 20190515054726_rnn_no4_abcdef_10_3.zip
# 20190515063038_rnn_no3_abcd_10_3.zip
# 20190513094540_rnn_no4_abcd_10_3.zip
# 20190513121140_rnn_no2_abcd_10_3.zip
# 20190513135055_rnn_no1_abcdef_10_3.zip
# 20190513135613_rnn_no1_abcd_10_3.zip
# 20190515060412_rnn_no3_abcdefghij_10_3.zip
# 20190515084348_rnn_no2_abcdefghij_20_3.zip
# 20190515084609_rnn_no2_abcdefghij_10_3.zip
# 20190515112429_rnn_no1_abcdefghijklmno_10_3.zip
# 20190514142823_rnn_no4_abcdefghij_20_3.zip
# 20190515084307_rnn_no2_abcdefghijklmno_10_3.zip
# 20190514171225_rnn_no3_abcdefghijklmno_10_3.zip
# 20190514091252_rnn_no5_abcdefghijklmno_20_3.zip
# 20190514091232_rnn_no5_abcdefghijklmno_15_3.zip
# 20190514142922_rnn_no4_abcdefghijklmno_10_3.zip
# 20190514170722_rnn_no3_abcdefghijklmno_20_3.zip
# 20190514143253_rnn_no4_abcdefghij_10_3.zip
# 20190515112326_rnn_no1_abcdefghijklmno_15_3.zip
# 20190515084323_rnn_no2_abcdefghijklmno_20_3.zip
# 20190515084246_rnn_no2_abcdefghijklmno_15_3.zip
# 20190515060703_rnn_no3_abcdefghij_20_3.zip
# 20190514142303_rnn_no4_abcdefghijklmno_15_3.zip
# 20190514114707_rnn_no5_abcdefghij_15_3.zip
# 20190514114621_rnn_no5_abcdefghij_20_3.zip
# 20190515112145_rnn_no1_abcdefghijklmno_20_3.zip
# 20190514091009_rnn_no5_abcdefghijklmno_10_3.zip
# 20190515060653_rnn_no3_abcdefghij_15_3.zip
# 20190514115036_rnn_no5_abcdefghij_10_3.zip
# 20190514142246_rnn_no4_abcdefghijklmno_20_3.zip
# 20190515131517_rnn_no1_abcdefghij_10_3.zip
# 20190514170952_rnn_no3_abcdefghijklmno_15_3.zip
# 20190514143034_rnn_no4_abcdefghij_15_3.zip
# 20190515084532_rnn_no2_abcdefghij_15_3.zip
# 20190515135758_rnn_no1_abcdefghij_20_3.zip
# 20190515135922_rnn_no1_abcdefghij_15_3.zip"""
# rnns = """20190827110441-172-31-28-104_rnn_no1_abcdefghijklmno_10_3.zip
# 20190827125600-172-31-27-60_rnn_no4_abcd_10_3.zip
# 20190827085359-172-31-27-42_rnn_no4_abcdefghij_15_3.zip
# 20190827125514-172-31-20-198_rnn_no3_abcd_10_3.zip
# 20190827090320-172-31-27-60_rnn_no3_abcdefghijklmno_20_3.zip
# 20190827090310-172-31-18-182_rnn_no3_abcdefghijklmno_15_3.zip
# 20190827070625-172-31-29-101_rnn_no4_abcdefghijklmno_10_3.zip
# 20190827130614-172-31-29-75_rnn_no1_abcd_10_3.zip
# 20190827070655-172-31-18-182_rnn_no5_abcdefghijklmno_15_3.zip
# 20190827090330-172-31-25-112_rnn_no3_abcdefghijklmno_10_3.zip
# 20190904144447-192-168-121-1_rnn_no3_abcdef_10_3.zip
# 20190827090530-172-31-20-198_rnn_no3_abcdefghij_15_3.zip
# 20190827125536-172-31-18-182_rnn_no4_abcdef_10_3.zip
# 20190827105954-172-31-27-60_rnn_no2_abcdefghij_10_3.zip
# 20190827130150-172-31-17-80_rnn_no1_abcdef_10_3.zip
# 20190827130301-172-31-28-104_rnn_no2_abcd_10_3.zip
# 20190827110040-172-31-20-198_rnn_no1_abcdefghijklmno_15_3.zip
# 20190827090911-172-31-17-80_rnn_no2_abcdefghijklmno_15_3.zip
# 20190827070707-172-31-25-112_rnn_no4_abcdefghij_20_3.zip
# 20190827125343-172-31-29-101_rnn_no5_abcd_10_3.zip
# 20190827070217-172-31-27-42_rnn_no4_abcdefghijklmno_20_3.zip
# 20190827104555-172-31-27-42_rnn_no2_abcdefghijklmno_10_3.zip
# 20190827071125-172-31-17-80_rnn_no5_abcdefghij_15_3.zip
# 20190827090650-172-31-23-42_rnn_no2_abcdefghijklmno_20_3.zip
# 20190827090609-172-31-28-104_rnn_no3_abcdefghij_20_3.zip
# 20190827090831-172-31-29-75_rnn_no3_abcdefghij_10_3.zip
# 20190827105956-172-31-25-112_rnn_no1_abcdefghijklmno_20_3.zip
# 20190827070825-172-31-28-104_rnn_no5_abcdefghij_10_3.zip
# 20190827105920-172-31-18-182_rnn_no2_abcdefghij_15_3.zip
# 20190827071031-172-31-23-42_rnn_no4_abcdefghijklmno_15_3.zip
# 20190827070909-172-31-20-198_rnn_no5_abcdefghijklmno_10_3.zip
# 20190827105821-172-31-29-101_rnn_no2_abcdefghij_20_3.zip
# 20190827090218-172-31-29-101_rnn_no4_abcdefghij_10_3.zip
# 20190827125741-172-31-23-42_rnn_no2_abcdef_10_3.zip
# 20190827110535-172-31-17-80_rnn_no1_abcdefghij_10_3.zip
# 20190827070645-172-31-27-60_rnn_no5_abcdefghijklmno_20_3.zip
# 20190827110218-172-31-23-42_rnn_no1_abcdefghij_20_3.zip
# 20190827123757-172-31-27-42_rnn_no5_abcdef_10_3.zip
# 20190827070915-172-31-29-75_rnn_no5_abcdefghij_20_3.zip
# 20190827110716-172-31-29-75_rnn_no1_abcdefghij_15_3.zip"""
rnns = "20190825235324-192-168-121-1_rnn_paren_42_10.zip"

rnns = rnns.split("\n")
rnns = [r.strip() for r in rnns]

print(len(set(rnns)), len(rnns))
input()
rows = []

def split_line(x):
    splitted = x.split()
    # print(splitted)
    if len(splitted) == 1:
        return "", float(splitted[0])
    elif len(splitted) == 2:
        return splitted[0], float(splitted[1])
    else:
        assert False

xs = []
for e in rnns:
    tf.reset_default_graph()
    flag_paren = ("paren" in e)

    print(e)
    util_boto3.download(e)
    task_name = "_".join(e.split("_")[2:])
    with zipfile.ZipFile(e) as z:
        dirname = os.path.splitext(e)[0]
        z.extractall(dirname)
    if flag_paren:
        alphabet = "()0123456789"
        wfa_orig = WFA.WFA("a", np.array([[0]]), np.array([[0]]), {"a": np.array([[0]])})
        wfa_orig.callings = set()
        alphabet_size = len(alphabet)
        alphabet2id, max_length, regr, words = precalc_rnn.set_for_paren(e, dirname)
    else:
        with open(os.path.join(dirname, "wfa.pickle"), "rb") as f:
            wfa_orig: WFA.WFA = pickle.load(f)
            wfa_orig.callings = set()
            alphabet = wfa_orig.alphabet
            states_orig = wfa_orig.get_size()
        alphabet2id, max_length, regr, words = precalc_rnn.set_for_artificial(e, dirname)

    with open(os.path.join(dirname, "args.json"), "r") as f:
        j = json.load(f)
        batch_size = j["batch_size"]

    if flag_paren:
        with open(os.path.join(dirname, "test.txt"), "r") as f:
            words_test = [split_line(x) for x in f.readlines()]
            words_test = [(Dataset.word2vec(x, alphabet2id, max_length), y) for x, y in words_test]
    else:
        with open(os.path.join(dirname, "test.txt"), "r") as f:
            words_test = []
            for l in f.readlines():
                w = l.strip()
                words_test.append((Dataset.word2vec(w, alphabet2id, max_length), wfa_orig.classify_word(w)))
    print("Testing: ")
    for i in words_test[:5]:
        print(f"{i[0]} : {i[1]}")
    for i in words_test:
        assert len(i[0]) <= max_length


    print("Starting evaluation")
    ev = eval_rnn.evaluate(regr, max_length, batch_size, words_test, dirname, lambda p: eval_rnn.eps_equality_absolute(p[0], p[1], 0.05))
    xs.append((e, ev.n_equals, ev.ave_loss))
    rows.append({"task name": task_name, "alphabet size": len(alphabet), "original wfa's size": wfa_orig.get_size(),
                 "mse": ev.ave_loss, "acc005": ev.n_equals/1000, "file": e})

    print(xs[-1])

with open("wrapup_rnn.csv", "w") as f:
    fieldnames = ["task name", "alphabet size", "original wfa's size",
                  "mse", "acc005", "file"]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

with open("wrapup_rnn.csv", "r") as f:
    x = "".join(f.readlines())
    x = x.replace("\n\n", "\n")

with open("wrapup_rnn.csv", "w") as f:
    f.write(x)
