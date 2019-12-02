import argparse
import os
import random
import util_boto3
import json
from util import sample_length_from_all_words, sample_length_from_all_lengths, make_words


def ordword(w):
    nums = []
    for i in w:
        nums.append(str(ord(i)))
    return ",".join(nums)


def mysort(w):
    alphabet = list(set(w))
    cnt = {a: w.count(a) for a in alphabet}
    random.shuffle(alphabet)
    return "".join(a * cnt[a] for a in alphabet)


def main():
    parser = argparse.ArgumentParser(
        description='Make a word dataset')
    parser.add_argument('alphabets',
                        help='The alphabet set')
    parser.add_argument('n_samples', type=int,
                        help='The size of the dataset')
    parser.add_argument('--max_length', type=int, default=20,
                        help='The maximum length of a word')
    parser.add_argument('--length_sampling', default='all_lengths',
                        choices=['all_words', 'all_lengths'],
                        help=('Samples the length of a word from '
                              'all_words or all_lengths'))
    parser.add_argument("--s3", action="store_true",
                        help="upload to S3")
    parser.add_argument("-o", type=str, default="result_make_word_dataset.zip",
                        help="filename of output")
    parser.add_argument("--alphabet_num", default=0, type=int,
                        help="num")
    parser.add_argument("--sorted", action="store_true", help="sorted")
    args = parser.parse_args()

    length_sampling = \
        sample_length_from_all_words if args.length_sampling == 'all_words' \
            else sample_length_from_all_lengths

    if args.alphabet_num > 0:
        print("replacing")
        args.alphabets = "".join([chr(i) for i in range(args.alphabet_num)])
    words = make_words(args.alphabets,
                       args.max_length,
                       args.n_samples,
                       length_sampling)
    print("sorted", args.sorted)
    if args.sorted:
        print(args.sorted)
        words = [mysort(w) for w in words]


    words_list = list(words)
    if args.alphabet_num > 0:
        words_list = [ordword(i) for i in words_list]
    random.shuffle(words_list)
    num_test = len(words_list) // 10
    words_test = words_list[:num_test]
    words_train = words_list[num_test:]
    with open("train.txt", "w") as f:
        # write(f, words_train, args.alphabets > 0)
        f.writelines("\n".join(words_train))
    with open("test.txt", "w") as f:
        # write(f, words_test, args.alphabets > 0)
        f.writelines("\n".join(words_test))
    with open("args.json", "w") as f:
        json.dump(vars(args), f)

    fn = util_boto3.zip_save(args.o, ["train.txt", "test.txt", "args.json"], True)
    if args.s3:
        print("upload")
        util_boto3.upload(fn)
    print(fn)

if __name__ == '__main__':
    main()
