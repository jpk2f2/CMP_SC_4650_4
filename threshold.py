import numpy as np


def find_grey_avg(im: np.ndarray):
    dimensions = im.shape
    total = 0
    count = 0
    avg = np.sum(im)/(dimensions[0]*dimensions[1])

    return int(avg)


def find_global_threshold(im: np.ndarray, diff: int, t: int):
    t_prev = t
    return t_prev


def global_threshold(im: np.ndarray, t: int):
    with np.nditer(im, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = 1 if x[...] > t else 0

    return im
