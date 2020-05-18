import cv2
import numpy as np


def dilation(B, S):
    ###############################################
    # TODO                                        #
    # dilation 함수 완성                          #
    ###############################################
    h, w = B.shape
    s_h, s_w = S.shape
    c_h = (int)(s_h / 2)
    c_w = (int)(s_w / 2)
    img_dilation = np.zeros((h, w), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            if S[c_h][c_w] == B[row][col]:
                for s_row in range(s_h):  # 필터돌기
                    for s_col in range(s_w):
                        i = max(min(row - c_h + s_row, h - 1), 0)
                        j = max(min(col - c_w + s_col, w - 1), 0)
                        img_dilation[i][j] = S[s_row][s_col]

    return img_dilation


def erosion(B, S):
    ##############################################
    # TODO                                        #
    # erosion 함수 완성                           #
    ###############################################
    h, w = B.shape
    s_h, s_w = S.shape
    c_h = (int)(s_h / 2)
    c_w = (int)(s_w / 2)
    img_erosion = np.zeros((h, w), dtype=np.uint8)

    check = 0

    for row in range(h):
        for col in range(w):
            if S[c_h][c_w] == B[row][col]:

                check = 1
                for s_row in range(s_h):  # 필터돌기
                    for s_col in range(s_w):
                        i = row - c_h + s_row
                        j = col - c_w + s_col
                        if i < 0 or j < 0 or i >= h or j >= w:
                            check = 0
                            continue

                        if B[i][j] != S[s_row][s_col]:
                            check = 0

            if check == 1:
                img_erosion[row][col] = 255
    return img_erosion


def opening(B, S):
    ###############################################
    # TODO                                        #
    # opening 함수 완성                           #
    ###############################################
    h, w = B.shape
    img_opening = erosion(B, S)
    img_opening = dilation(img_opening, S)

    return img_opening


def closing(B, S):
    ###############################################
    # TODO                                        #
    # closing 함수 완성                           #
    ###############################################
    h, w = B.shape
    img_closing = dilation(B, S)
    img_closing = erosion(img_closing, S)

    return img_closing


if __name__ == '__main__':
    B = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [255, 255, 255, 255, 255, 255, 255, 0],
         [0, 0, 0, 255, 255, 255, 255, 0],
         [0, 0, 0, 255, 255, 255, 255, 0],
         [0, 0, 255, 255, 255, 255, 255, 0],
         [0, 0, 0, 255, 255, 255, 255, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]]
        , dtype=np.uint8)

    S = np.array(
        [[255, 255, 255],
         [255, 255, 255],
         [255, 255, 255]]
        , dtype=np.uint8)

    cv2.imwrite('morphology_B.png', B)

    img_dilation = dilation(B, S)
    print(img_dilation)
    cv2.imwrite('morphology_dilation.png', img_dilation)

    img_erosion = erosion(B, S)
    print(img_erosion)
    cv2.imwrite('morphology_erosion.png', img_erosion)

    img_opening = opening(B, S)
    print(img_opening)
    cv2.imwrite('morphology_opening.png', img_opening)

    img_closing = closing(B, S)
    print(img_closing)
    cv2.imwrite('morphology_closing.png', img_closing)
