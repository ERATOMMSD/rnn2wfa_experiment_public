from typing import *

import numpy as np

import WFA


def word2vec(word: str,
             alphabet2id: Dict[str, int],
             max_length: int) -> np.ndarray:
    # if len(word) > max_length:
    #     raise ValueError(
    #         f'the length of word "{word}" is more than {max_length}')
    vec = [alphabet2id[c] for c in word]
    vec += [0.] * (max_length - len(vec))
    # assert len(vec) == max_length
    return np.array(vec)


def make_alphabet2id_dict(alphabets: str) -> Dict[str, int]:
    return {a: i + 1 for i, a in enumerate(alphabets)}


def load_data(file_path: str,
              wfa: WFA,
              alphabet2id: Dict[str, int],
              max_length: int) -> List[Tuple[np.ndarray, float]]:
    with open(file_path, 'r') as f:
        words = [w.strip() for w in f]
    return [(word2vec(w, alphabet2id, max_length), wfa.classify_word(w))
            for w in words]


def parse_alphabet_tsv(path: str) -> str:
    with open(path, 'r') as f:
        # sum is the flatten in python3
        return ''.join(sum([x.split() for x in f.readlines()], []))


def load_data_tsv(file_path: str,
                  alphabet2id: Dict[str, int],
                  max_length: int) -> List[Tuple[np.ndarray, float]]:
    with open(file_path, 'r') as f:
        words = [line.split() for line in f.readlines()]
    return [(word2vec(''.join(w[:-2]), alphabet2id, max_length), float(w[-1]))
            for w in words]
