import numpy as np
import scipy.signal
import cv2
import math


# preprocesses given image, returns padded version and array w/ original dimensions for final image
# takes in image, desired padding, and pad type such as 'zero' or 'repeat'
def prepare_image(im: np.ndarray, padding: int, pad_type: str):
    dimensions = im.shape  # get dimensions of original image for creating new image
    # convert brg to grayscale
    if (len(dimensions)) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # create new image w/ appropriate dimensions
    im2 = np.zeros((dimensions[0], dimensions[1]))

    # use specified padding
    if pad_type == 'zero':
        # pad with zeros
        im = np.pad(im, padding, 'constant', constant_values=0)
    elif pad_type == 'repeat':
        # copy nextdoor pixel for padding
        im = np.pad(im, padding, 'symmetric')
    else:
        print('This should not have been reached')

    return im.astype(dtype=np.float32), im2.astype(dtype=np.float32)  # return processed image


# post processes given image
# takes in image and whether or not to convert to rgb
def pp_image(im: np.ndarray, g2rgb: bool = False) -> np.ndarray:
    if g2rgb:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    # return pp image with correct type
    return im.astype(dtype='uint8')


# creates a gaussian matrix using a given kernel size for convulation w/ an image
# returns the guassian matrix
# for example, a kernel size of 2 creates a 5x5 gaussian array
def create_gauss_conv(kernel: int) -> np.ndarray:
    # create array to hold guassian array
    final = np.zeros(shape=(2 * kernel + 1, 2 * kernel + 1))
    tot = 0
    # loop through array, creating gaussian distribution
    for i in range(-kernel, kernel + 1):
        for j in range(-kernel, kernel + 1):
            # create gaussian dividend
            tmp = -1 * ((i ** 2 + j ** 2) / (2 * (kernel ** 2)))
            # complete gaussian function and place it in dest
            final[i + kernel, j + kernel] = math.exp(tmp) / (2 * np.pi * kernel ** 2)
            # count total for normalization
            tot = tot + final[i + kernel, j + kernel]

    # normalize gaussian array
    final = final / tot

    return final


def gauss_filter(im: np.ndarray, kernel: int) -> np.ndarray:
    # check if premade desired
    _filter = create_gauss_conv(kernel)  # create a guassian filter for the given kernel

    im, im_proc = prepare_image(im, kernel, 'zero')  # preprocess image
    # convolve image with filter
    # 'valid' specified to use preprocessed padding
    im_proc = scipy.signal.convolve2d(im, _filter, 'valid')

    # post process image and return it
    return pp_image(im_proc, False)


def canny_display(ims, sig, t_h, t_l):
    titles = ('original', 'gauss filtered w/ sigma: {}'.format(sig), 'Fx', 'Fy', 'F', 'D', 'Nonmax Suppression',
              'Hysterisis w/ t_h: {} and t_l: {}'.format(t_h, t_l))
    for i in range(0, len(ims)):
        cv2.imshow(titles[i], cv2.normalize(ims[i], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def canny_write(ims, sig, t_h, t_l, im_name):
    titles = ('original', 'gauss sigma', 'Fx', 'Fy', 'F', 'D', 'Nonmax Suppression',
              'Hysterisis')
    for i in range(0, len(ims)):
        cv2.imwrite('resources/plots/{}_{}_{}-{}-{}.png'.format(im_name, titles[i], sig, t_h, t_l),
                    cv2.normalize(ims[i], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1))
    return

