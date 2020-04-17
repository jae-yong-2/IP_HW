import cv2
import numpy as np
import matplotlib.pyplot as plt


def my_average_filter_3x3(src):
    mask = -(1 / 9) * np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]]) \
           + \
           3 * np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]])

    dst = cv2.filter2D(src, -1, mask)
    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    dst = my_average_filter_3x3(src)

    cv2.imshow('original', src)
    cv2.imshow('average filter', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
