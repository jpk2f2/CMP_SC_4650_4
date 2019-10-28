import cv2
import matplotlib
import utility
import threshold

matplotlib.use('TkAgg')

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

im_test = threshold.global_threshold_apply(im_1001a, 2)
cv2.imshow('test', cv2.normalize(im_test, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1))
cv2.waitKey(0)