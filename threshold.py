import numpy as np
import random


# finds average gray value of entire imagae
# takes in image array
# returns average grey value
def find_grey_avg(im: np.ndarray):
    dimensions = im.shape  # get size of image
    avg = np.sum(im) / (dimensions[0] * dimensions[1])  # sum image values and divide by total

    return int(avg)


# find global threshold value T of a given image
# takes in an image array, minimum difference in successive thresholds, and an intial threshold
# recommend using find_grey_avg() for the initial threshold t
# returns the global threshold
def find_global_threshold(im: np.ndarray, diff: int, t: int):
    # initialize counters/value holders
    t_prev = t
    count1 = 0
    count2 = 0
    sum1 = 0
    sum2 = 0

    # iterate image array
    # split on current threshold and average seperately
    with np.nditer(im, op_flags=['readonly']) as it:
        for x in it:
            if x[...] > t:
                count1 += 1
                sum1 += x[...]
            else:
                count2 += 1
                sum2 += x[...]

    t = int(((sum1 / count1 + sum2 / count2) / 2))  # recombine and normalize
    # check if less than predefined difference
    if abs(t - t_prev) < diff:
        return t  # break recursion
    else:
        return find_global_threshold(im, diff, t)  # recursion


# sets image values based on threshold value
# takes in image array and threshold T
# returns processed image, all values 0 or 1
def global_threshold(im: np.ndarray, t: int):
    im2 = im.copy()  # prevent altering original image
    # iterate image array and change values
    with np.nditer(im2, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = 1 if x[...] > t else 0

    return im2


# applies all functions necessary to conduct complete Basic Global Thresholding algorithm
# takes in image array and difference for use in finding the global threshold value
def global_threshold_apply(im: np.ndarray, diff: int):
    t = find_global_threshold(im, diff, find_grey_avg(im))  # find global threshold using avg grey value to start

    # apply global threshold and return
    return global_threshold(im, t)


# K-Means Clustering algorithm
# Segments image based on K number of cluster
# takes in an image array, K number of clusters, and the type of cluster initialization
def k_means(im: np.ndarray, k: int, initial: str):
    # get max and min values of image for cluster initialization
    max_val = np.max(im)
    min_val = np.min(im)
    dimensions = im.shape  # get size of image
    total = dimensions[0] * dimensions[1]
    im_flat = im.flatten()  # flatten image for processing
    # initalize clusters
    centroids = np.zeros([k])
    centroids_prev = np.zeros([k])

    if initial == 'rand':  # randomly select values within the image range
        centroids = random.sample(range(min_val, max_val+1), k)
    elif initial == 'spaced':  # select K evenly spaced values within the image range
        idx = np.round(np.linspace(min_val, max_val, k)).astype(int)
        centroids = idx

    # holds cluster association for each pixel of image
    cluster_array = np.zeros([total], dtype='int8')

    # loop until cluster centers stop moving
    while not np.array_equiv(centroids, centroids_prev):
        centroids_prev = centroids

        # calculate distance between each point and each cluster center
        # keep minimum value for each
        for i in range(im_flat.size):
            dist_array = np.zeros([k])
            for j in range(k):
                dist_array[j] = np.linalg.norm(im_flat[i]-centroids[j])
            cluster_array[i] = int(np.argmin(dist_array))

        # set new cluster centers based on associated pixels
        for i in range(k):
            count = 0
            summ = 0
            for j in range(cluster_array.size):
                if i == cluster_array[j]:
                    count += 1
                    summ += im_flat[j]
                centroids[i] = int(summ/count) if count > 0 else centroids[i]
    # assign pixel values to associated cluster center values
    for i in range(k):
        for j in range(im_flat.size):
            if cluster_array[j] == i:
                # print('here')
                im_flat[j] = centroids[i]

    # reshape and return processed image array
    return im_flat.reshape([dimensions[0], dimensions[1]])
