import cv2
import matplotlib
import utility
import threshold

from matplotlib import pyplot as plt

matplotlib.use('TkAgg')

# read in example images as grayscale
im_1001a = cv2.imread('resources/Fig1001(a)(constant_gray_region).tif', 0)
im_1001d = cv2.imread('resources/Fig1001(d)(noisy_region).tif', 0)
im_1022a = cv2.imread('resources/Fig1022(a)(building_original).tif', 0)
im_1026a = cv2.imread('resources/Fig1026(a)(headCT-Vandy).tif', 0)
im_1027a = cv2.imread('resources/Fig1027(a)(van_original).tif', 0)
im_1030a = cv2.imread('resources/Fig1030(a)(tooth).tif', 0)
im_1034a = cv2.imread('resources/Fig1034(a)(marion_airport).tif', 0)
im_1043a = cv2.imread('resources/Fig1043(a)(yeast_USC).tif', 0)
im_1045a = cv2.imread('resources/Fig1045(a)(iceberg).tif', 0)
im_1060a = cv2.imread('resources/Fig1060(a)(car on left).tif', 0)

# group example images and prepare names
images = [im_1001a, im_1001d, im_1022a, im_1026a, im_1027a, im_1030a, im_1034a, im_1043a, im_1045a, im_1060a]
titles = ['im_1001a', 'im_1001d', 'im_1022a', 'im_1026a', 'im_1027a', 'im_1030a', 'im_1034a', 'im_1043a',
          'im_1045a', 'im_1060a']


# for presentation
# apply global thresholding on list of images
# takes in list of images, and difference for global thresholding
# returns list of original and processed images
def apply_global_thresholding(ims: list, diff: int):
    ims_processed = []  # hold original and completed images
    for im in ims:
        print('Processing image w/ global thresholding')  # track image progress
        ims_processed.append(im)  # append original image
        ims_processed.append(threshold.global_threshold_apply(im, 2))  # process and append new image

    print('Completed all images\n')
    return ims_processed


# for presentation
# display list of images, one set at a time
# takes in list of images for display, list of display names, size of each set, and type of alg used on input images
def display_imageset(ims: list, names: list, step: int, alg: str):
    i = 0
    j = 0
    names_global = ['original', 'global thresholded']
    names_k = ['original', 'k = 3', 'k = 5']
    for im in ims:
        if alg == 'global':
            if i % step == 0:
                cv2.imshow(names[j] + ' ' + names_global[i % step], im)
            else:
                # normalize global thresholded image
                cv2.imshow(names[j] + ' ' + names_global[i % step],
                           cv2.normalize(im, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1))
        else:
            cv2.imshow(names[j] + ' ' + names_k[i % step], im)
        i += 1
        if i % step == 0:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            j += 1

    return


# for presentation
# apply K-Mean Clustering algorithm on given list of images
# takes in list of input images, type of initial cluster allocation, and number of cluster
# k = 0 runs the algorithm twice with both a k of 3 and a k of 5
# returns a list of original image - processed image sets
def apply_k_means(ims: list, initial: str, k: int = 0):
    ims_processed = []  # holds original and processed images
    if k == 0:
        for im in ims:
            print('Processing image w/ k-means')  # track progress
            ims_processed.append(im)
            ims_processed.append(threshold.k_means(im, 3, initial))
            ims_processed.append(threshold.k_means(im, 5, initial))
    else:
        for im in ims:
            ims_processed.append(im)
            ims_processed.append(threshold.k_means(im, k, initial))

    print('Completed all images\n')
    return ims_processed


# writes images given in list to 'plots' folder under ' resources'
# takes in list of images to be written, names of images, size of image sets, and type of alg run on input images
def write_imageset(ims: list, names: list, step: int, alg: str):
    i = 0
    j = 0
    names_global = ['original', 'global thresholded']
    names_k = ['original', 'k = 3', 'k = 5']
    for im in ims:
        if alg == 'global':
            if i % step == 0:
                cv2.imwrite('resources/plots/{}_{}.png'.format(names[j], names_global[i % step]), im)
            else:
                # normalize global thresholded image
                cv2.imwrite('resources/plots/{}_{}.png'.format(names[j], names_global[i % step]),
                            cv2.normalize(im, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1))
        else:
            cv2.imwrite('resources/plots/{}_{}.png'.format(names[j], names_k[i % step]), im)
        i += 1
        if i % step == 0:
            j += 1

    return


# the commented block below is for generating histograms on the k means clustering algorithm
'''
plt.hist(im_1060a.ravel(), 256, [0, 256])
plt.show()
im_test = threshold.k_means(im_1060a, 3, 'spaced')
plt.hist(im_test.ravel(), 256, [0, 256])
plt.show()
im_test2 = threshold.k_means(im_1060a, 5, 'spaced')
plt.hist(im_test2.ravel(), 256, [0, 256])
plt.show()
'''


# run k cluster algorithm on image set
# display processed images next to their original counterparts
# display_imageset(apply_k_means(images, 'spaced'), titles, 3, 'k-means')
ims_k = apply_k_means(images, 'spaced')
display_imageset(ims_k, titles, 3, 'k-means')
write_imageset(ims_k, titles, 3, 'k-means')

# run global threshold algorithm on image set
# display processed images next to their original counterparts
# display_imageset(apply_global_thresholding(images, 2), titles, 2, 'global')
ims_global = apply_global_thresholding(images, 2)
display_imageset(ims_global, titles, 2, 'global')
write_imageset(ims_global, titles, 2, 'global')


# old code below, kept in case needed for presenting

# im_test = threshold.global_threshold_apply(im_1001a, 2)
# im_test = threshold.k_means_intial(im_1060a, 3, 'spaced')
# im_test = threshold.k_means_intial(im_1001a, 1)
# cv2.imshow('test', cv2.normalize(im_test, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1))
# cv2.imshow('test', im_test)
# cv2.waitKey(0)
