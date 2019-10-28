import numpy as np


def find_grey_avg(im: np.ndarray):
    dimensions = im.shape
    avg = np.sum(im) / (dimensions[0] * dimensions[1])

    return int(avg)


def find_global_threshold(im: np.ndarray, diff: int, t: int):
    t_prev = t
    count1 = 0
    count2 = 0
    sum1 = 0
    sum2 = 0

    with np.nditer(im, op_flags=['readonly']) as it:
        for x in it:
            if x[...] > t:
                count1 += 1
                sum1 += x[...]
            else:
                count2 += 1
                sum2 += x[...]

    t = int(((sum1 / count1 + sum2 / count2) / 2))
    if abs(t - t_prev) < diff:
        return t
    else:
        return find_global_threshold(im, diff, t)


def global_threshold(im: np.ndarray, t: int):
    with np.nditer(im, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = 1 if x[...] > t else 0

    return im


def global_threshold_apply(im: np.ndarray, diff: int):
    t = find_global_threshold(im, diff, find_grey_avg(im))

    return global_threshold(im, t)
