"""
General classes, functions, utilities that are used throughout rlkit.
"""

from typing import List
from collections import defaultdict
import numpy as np


def list_dict_to_dict_list(list_dict):
    dict_list = defaultdict(list)
    for _dict in list_dict:
        for k, v in _dict.items():
            dict_list[k].append(v)
    return dict(dict_list)


def dict_list_to_list_dict(dict_list):
    # For example,
    # Input: {"agent_0": [1, 2], "agent_1": [3]}
    # Output: [{"agent_0": 1, "agent_1": 3}, {"agent_0": 2}]
    # support not equal length list
    maxlen = max([len(v) for v in dict_list.values()])
    list_dict = [{} for _ in range(maxlen)]
    for k, v in dict_list.items():
        for idx in range(len(v)):
            list_dict[idx][k] = v[idx]
    return list_dict


# integer division
def split_integer(integer: int, n_parts: int, mode: str = "equal") -> List[int]:
    assert integer > 0 and n_parts > 0, f"INT: {integer}, N_PARTS:{n_parts}"
    if n_parts == 1:
        return [integer]
    if mode == "equal":
        quotient = int(integer / n_parts)
        remainder = integer % n_parts
        if remainder > 0:
            return [quotient] * (n_parts - remainder) + [quotient + 1] * remainder
        if remainder < 0:
            return [quotient - 1] * -remainder + [quotient] * (n_parts + remainder)
        return [quotient] * n_parts

    elif mode == "random":
        split_point = np.random.randint(integer, size=n_parts - 1)
