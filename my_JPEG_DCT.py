import cv2
import numpy as np


def my_normalize(src):
    dst = src.copy()
    if np.min(dst) != np.max(dst):
        dst = dst - np.min(dst)
    dst = dst / np.max(dst) * 255
    return dst.astype(np.uint8)


def my_DCT(src, n=8):
    ###############################
    # TODO                        #
    # my_DCT 완성                 #
    # src : input image           #
    # n : block size              #
    ###############################
    (h, w) = src.shape
    dct_img = (src.copy()).astype(np.float)
    dst = np.zeros((h, w))
    mask = np.zeros((n, n), dtype=np.float)
    C_left = 0
    C_right = 0
    for row_num in range(h // n):
        for col_num in range(w // n):


            for block_row in range(n):
                for block_col in range(n):
                    if row_num == 0:
                        C_left = np.sqrt(1 / n)
                    elif row_num != 0:
                        C_left = np.sqrt(2 / n)

                    if col_num == 0:
                        C_right = np.sqrt(1 / n)
                    elif col_num != 0:
                        C_right = np.sqrt(2 / n)

                    for row in range(n):
                        for col in range(n):
                            mask[row][col] = np.cos((2 * row + 1) * np.pi * block_row / (2 * n)) * np.cos(
                                (2 * col + 1) * np.pi * block_col / (2 * n))
                    dst[row_num*n + block_col][col_num*n + block_row] = C_left*C_right*np.sum(
                        dct_img[row_num*n: (row_num+1)*n, col_num*n: (col_num+1)*n] * mask)

    return my_normalize(dst)


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    dst = my_DCT(src, 8)

    dst = my_normalize(dst)
    cv2.imshow('my DCT', dst)

    cv2.waitKey()
    cv2.destroyAllWindows()
