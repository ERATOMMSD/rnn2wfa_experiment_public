from typing import *
import requests
import json
import datetime
import random
import socket
import numpy as np

try:
    with open("slack_webhook.txt", "r") as f:
        slack_url = f.readline().strip()
except Exception as e:
    pass


def bfs_words(alphabet: str, limit_depth: Optional[int] = None, limit_num: Optional[int] = None) -> Iterable[str]:
    num = 0
    queue = [""]
    while True:
        word = queue.pop(0)
        if limit_depth is not None and len(word) > limit_depth:
            break
        num += 1
        yield word
        if limit_num is not None and num >= limit_num:
            break
        queue += [word + c for c in alphabet]


def notify_slack(message: str) -> bool:
    try:
        requests.post(slack_url, data=json.dumps({
            'text': message,  # 投稿するテキスト
            'username': u'rnn2wfa',  # 投稿のユーザー名
            'icon_emoji': u':ghost:',  # 投稿のプロフィール画像に入れる絵文字
            'link_names': 1,  # メンションを有効にする
        }))
        return True
    except Exception as e:
        return False


def get_time_hash():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "-" \
           + socket.gethostbyname(socket.gethostname()).replace(".", "-")


T = TypeVar("T")


def weighted_choice(x: Dict[T, int]) -> T:
    """
    Equivalent to random.choice(concat([[k]*v for k, v in x.items()]))
    :param x:
    :return:
    """
    n = sum(x.values())
    val = random.randint(0, n - 1)
    for k, v in x.items():
        if val < v:
            return k
        val -= v
    assert False


def argmax_dict(d: Dict[T, float]) -> T:
    assert len(d) > 0
    maxk, maxv = None, None
    for k, v in d.items():
        if maxk is None or maxv < v:
            maxk = k
            maxv = v
    return maxk


def sample_length_from_all_words(n_alphabets: int,
                                 max_length: int) -> int:
    return weighted_choice({i: i ** n_alphabets for i in range(max_length + 1)})


def sample_length_from_all_lengths(n_alphabets: int,
                                   max_length: int) -> int:
    return random.randint(1, max_length)


def make_words(alphabets: str,
               max_length: int,
               n_samples: int,
               length_sampling: Callable[[int, int], int],
               exclude_list: Optional[List[str]] = None,
               random_seed: Optional[int] = None) -> List[str]:
    if exclude_list is None:
        exclude_list = []
    if random_seed is not None:
        rstate = random.getstate()
        random.seed(random_seed)
    n_alphabets = len(alphabets)
    abort_counter = 0
    words = set()
    while len(words) < n_samples:
        abort_counter += 1
        if abort_counter > n_samples * 100:
            break
        length = length_sampling(n_alphabets, max_length)
        word = ''
        for _ in range(length):
            word += random.choice(alphabets)
        assert len(word) <= max_length
        if word in exclude_list:
            continue
        words.add(word)
    if random_seed is not None:
        random.setstate(rstate)

    return list(words)


def dist_f(d: Callable[[np.ndarray, np.ndarray], float],
           f: Callable[[T], np.ndarray],
           x: T,
           y: T) -> float:
    return d(f(x), f(y))
