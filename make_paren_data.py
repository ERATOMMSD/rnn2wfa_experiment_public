import argparse
import os
import pickle
import random
from typing import *
import numpy as np
from WFA import WFA
import util_boto3
import json


def calc_depth(w):
    v = 0
    m = 0
    for c in w:
        if c == "(":
            v += 1
        elif c == ")":
            v -= 1
        else:
            pass
        if v < 0:
            return -1
        m = max(v, m)
    if v != 0:
        return -1
    return m


def make_balanced_word(n):
    length = 2 * n
    x = 0
    y = 0
    w = ""
    for i in range(length):
        if y == n:
            opt = [")"]
        elif x == y:
            opt = ["("]
        else:
            opt = ["(", ")"]
        c = random.choice(opt)
        if c == "(":
            y += 1
        elif c == ")":
            x += 1
        else:
            assert False
        w += c
        assert 0 <= x and x <= n
        assert 0 <= y and y <= n
    assert x == n and y == n
    return w


def calc_cost(w):
    d = calc_depth(w)
    if d == -1:
        return 0
    else:
        return 1 - 2 ** (-d)


def mutate_add(w):
    if w == "":
        return w
    n = random.randrange(0, len(w))
    return w[:n] + w[n] + w[n:]


def mutate_erase(w):
    if len(w) == 0:
        return w
    n = random.randrange(0, len(w))
    return w[:n] + w[n + 1:]


def mutate_swap(w):
    if len(w) <= 1:
        return w
    n = random.randrange(0, len(w) - 1)
    return w[:n] + w[n + 1] + w[n] + w[n + 2:]


def mutate(w, prob_mutate=0.5):
    while True:
        cs = [mutate_add, mutate_erase, mutate_swap]
        c = random.choice(cs)
        w = c(w)
        if random.random() > prob_mutate:
            break
    return w


def mix_words(a: str, b: str) -> str:
    xs = [0] * len(a) + [1] * len(b)
    random.shuffle(xs)
    w = ""
    ai = 0
    bi = 0
    for i in xs:
        if i == 0:
            w += a[ai]
            ai += 1
        elif i == 1:
            w += b[bi]
            bi += 1
        else:
            assert False
    return w


def make_word_positive(length, alphabet):
    n = random.randint(0, length // 2)
    paren_part = make_balanced_word(n)
    alph_part = "".join(random.choice(alphabet) for _ in range(length - len(paren_part)))
    return mix_words(paren_part, alph_part)


def make_word_negative(length, alphabet):
    n = random.randint(0, length // 2)
    paren_part = mutate(make_balanced_word(n), 0.5)
    alph_part = "".join(random.choice(alphabet) for _ in range(length - len(paren_part)))
    return mix_words(paren_part, alph_part)


def make_random_paren_words(alphabet, max_length, num):
    words = [make_word_positive(random.randint(0, max_length), alphabet)[:max_length] for _ in range(num // 2)]
    words += [make_word_negative(random.randint(0, max_length), alphabet)[:max_length] for _ in range(num // 2)]
    random.shuffle(words)
    return words


def main():
    parser = argparse.ArgumentParser(
        description='Make a probabilistic automaton')
    parser.add_argument('alphabet',
                        help='The alphabet sets')
    parser.add_argument("--s3", action="store_true",
                        help="upload to S3")
    parser.add_argument("-o", type=str, default="result_make_paren_data.zip",
                        help="filename of output")
    parser.add_argument("--max_length", default=20, type=int,
                        help="length")
    parser.add_argument("--num", default=1000, type=int,
                        help="num")
    parser.add_argument("--seed", default=42, type=int,
                        help="seed")
    args = parser.parse_args()

    random.seed(args.seed)

    words = make_random_paren_words(args.alphabet, args.max_length, args.num)

    for i, w in enumerate(words):
        print(w, calc_cost(w))

    num_test = len(words) // 10
    words_test = words[:num_test]
    words_train = words[num_test:]
    words_and_values_test = [f"{w}\t{calc_cost(w)}" for w in words_test]
    words_and_values_train = [f"{w}\t{calc_cost(w)}" for w in words_train]
    with open("train.txt", "w") as f:
        f.writelines("\n".join(words_and_values_train))
    with open("test.txt", "w") as f:
        f.writelines("\n".join(words_and_values_test))
    with open("args.json", "w") as f:
        json.dump(vars(args), f)
    with open("alphabet.txt", "w") as f:
        f.write("()" + args.alphabet)

    fn = util_boto3.zip_and_upload(args.o, ["train.txt", "test.txt", "args.json", "alphabet.txt"], True)
    if args.s3:
        util_boto3.upload(fn)

    print(fn)


if __name__ == '__main__':
    main()
