import subprocess
import itertools


def case1():
    alphs = ["ab", "abcd", "abcdef"]

    for i, alph in itertools.product(range(1, 11), alphs):
        cmd = f'python make_prob_automaton.py hoge "{alph}" 10 3 --s3="wfa_no{i}_{alph}_10_3.zip"'
        subprocess.run(cmd, shell=True)


def case2():
    abc = "abcdefghijklmnopqrstuvwxyz"
    alphs = [abc[:10], abc[:15], abc[:20]]
    sizes = [10, 15, 20]

    for i, alph, size in itertools.product(range(1, 5 + 1), alphs, sizes):
        cmd = f'python make_prob_automaton.py hoge "{alph}" {size} 3 --s3="wfa_no{i}_{alph}_{size}_3.zip"'
        subprocess.run(cmd, shell=True)


case2()
