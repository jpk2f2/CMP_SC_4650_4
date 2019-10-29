import numpy as np
import random


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
    im2 = im.copy()
    with np.nditer(im2, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = 1 if x[...] > t else 0

    return im2


def global_threshold_apply(im: np.ndarray, diff: int):
    t = find_global_threshold(im, diff, find_grey_avg(im))

    return global_threshold(im, t)


# placing this here to remember
# dist = numpy.linalg.norm(a-b)


def k_means(im: np.ndarray, k: int, initial: str):
    max_val = np.max(im)
    min_val = np.min(im)
    dimensions = im.shape
    total = dimensions[0] * dimensions[1]
    im_flat = im.flatten()
    centroids = np.zeros([k])
    centroids_prev = np.zeros([k])

    if initial == 'rand':
        centroids = random.sample(range(0, 256), k)
        # for i in range(k):
        #    rand = random.randint(0, total-1)
        #    centroids[i] = im_flat[rand]
    elif initial == 'spaced':
        idx = np.round(np.linspace(min_val, max_val, k)).astype(int)
        centroids = idx
        # idx = np.round(np.linspace(0, len(im_flat) - 1, k)).astype(int)
        # for i in range(k):
        #   print(idx[i])
        # centroids = im_flat[idx]

    cluster_array = np.zeros([total], dtype='int8')

    while not np.array_equiv(centroids, centroids_prev):
        centroids_prev = centroids
        for i in range(im_flat.size):
            dist_array = np.zeros([k])
            for j in range(k):
                dist_array[j] = np.linalg.norm(im_flat[i]-centroids[j])
            cluster_array[i] = int(np.argmin(dist_array))
            #print(cluster_array[i])

        for i in range(k):
            count = 0
            summ = 0
            for j in range(cluster_array.size):
                if k == cluster_array[j]:
                    count += 1
                    summ += im_flat[j]
                centroids[i] = int(summ/count) if count > 0 else centroids[i]

    for i in range(k):
        for j in range(im_flat.size):
            if cluster_array[j] == i:
                # print('here')
                im_flat[j] = centroids[i]

    return im_flat.reshape([dimensions[0], dimensions[1]])
