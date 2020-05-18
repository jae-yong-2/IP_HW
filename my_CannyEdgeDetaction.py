import cv2
import numpy as np


def my_padding(src, pad_shape, pad_type='zero'):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
    pad_img[p_h:h + p_h, p_w:w + p_w] = src

    if pad_type == 'repetition':
        # print('repetition padding')
        # up
        pad_img[:p_h, p_w:p_w + w] = src[0, :]
        # down
        pad_img[p_h + h:, p_w:p_w + w] = src[h - 1, :]

        # left
        pad_img[:, :p_w] = pad_img[:, p_w:p_w + 1]
        # right
        pad_img[:, p_w + w:] = pad_img[:, p_w + w - 1:p_w + w]

    else:
        # else is zero padding
        # print('zero padding')
        pass

    return pad_img


def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape

    # mask의 크기
    (m_h, m_w) = mask.shape

    # mask 확인
    # print('<mask>')
    # print(mask)

    # 직접 구현한 my_padding 함수를 이용
    pad_img = my_padding(src, (m_h // 2, m_w // 2), pad_type)

    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + m_h, col:col + m_w] * mask)
    return dst


# low-pass filter를 적용 후 high-pass filter적용
def apply_lowNhigh_pass_filter(src, fsize, sigma=1, pad_type='zero'):
    #########################################################################################
    # TODO                                                                                   #
    # apply_lowNhigh_pass_filter 완성                                                        #
    # Ix : image에 DoG_x filter 적용 or gaussian filter 적용된 이미지에 sobel_x filter 적용    #
    # Iy : image에 DoG_y filter 적용 or gaussian filter 적용된 이미지에 sobel_x filter 적용    #
    ###########################################################################################
    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering을 해도 되고, DoG를 이용해 한번에 해도 됨
    y, x = np.mgrid[-(fsize // 2):(fsize // 2) + 1, -(fsize // 2):(fsize // 2) + 1]
    DoG_x = (-x / sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    DoG_y = (-y / sigma ** 2) * np.exp(-(y ** 2 + y ** 2) / (2 * sigma ** 2))

    Ix = my_filtering(src, DoG_x, 'repetition')
    Iy = my_filtering(src, DoG_y, 'repetition')

    return Ix, Iy


# Ix와 Iy의 magnitude를 구함
def calcMagnitude(Ix, Iy):
    ##################################################
    # TODO                                           #
    # calcMagnitude 완성                             #
    # magnitude : ix와 iy의 magnitude를 계산         #
    #################################################
    magnitude = (Ix ** 2 + Iy ** 2) ** 0.5
    return magnitude


# Ix와 Iy의 angle을 구함
def calcAngle(Ix, Iy):
    #######################################
    # TODO                                #
    # calcAngle 완성                      #
    # angle     : ix와 iy의 angle         #
    #######################################

    angle = np.degrees(np.arctan2(Iy, Ix))

    return angle


# non-maximum supression 수행
def non_maximum_supression(magnitude, angle):
    ####################################################################################
    # TODO                                                                             #
    # non_maximum_supression 완성                                                      #
    # larger_magnitude     : non_maximum_supression 결과(가장 강한 edge만 남김)         #
    ####################################################################################
    (h, w) = magnitude.shape
    larger_magnitude = np.zeros((h, w))
    m1 = 0
    m2 = 0
    for row in range(w):
        for col in range(h):

            if 1 <= row < w - 1 and 1 <= col < h - 1:
                if 0 <= angle[row][col] <= 45 or -180 <= angle[row][col] <= -135:  # 1사분면 아래
                    t = np.tan(np.radians(angle[row][col]))
                    m1 = (1 - t) * magnitude[row][col + 1] + t * magnitude[row + 1][col + 1]
                    m2 = (1 - t) * magnitude[row][col - 1] + t * magnitude[row - 1][col - 1]

                elif 45 <= angle[row][col] <= 90 or -135 <= angle[row][col] <= -90:  # 1사분면 위
                    t = np.tan(np.radians(90 - angle[row][col]))
                    m1 = t * magnitude[row + 1][col + 1] + (1 - t) * magnitude[row + 1][col]
                    m2 = (1 - t) * magnitude[row - 1][col] + t * magnitude[row - 1][col - 1]

                elif 90 <= angle[row][col] <= 135 or -90 <= angle[row][col] <= -45:  # 2사분면 위
                    t = np.tan(np.radians(angle[row][col] - 90))
                    m1 = (1 - t) * magnitude[row + 1][col] + t * magnitude[row + 1][col - 1]
                    m2 = (1 - t) * magnitude[row - 1][col] + t * magnitude[row + 1][col + 1]

                elif 135 <= angle[row][col] <= 180 or -45 <= angle[row][col] <= 0:  # 2사분면 아래
                    t = np.tan(np.radians(180 - angle[row][col]))
                    m1 = t * magnitude[row + 1][col - 1] + (1 - t) * magnitude[row][col - 1]
                    m2 = (1 - t) * magnitude[row][col + 1] + t * magnitude[row - 1][col + 1]

            if magnitude[row][col] >= m1 and magnitude[row][col] >= m2 and 1 <= row < w - 1 and 1 <= col < h - 1:
                larger_magnitude[row][col] = magnitude[row][col]
            else:
                larger_magnitude[row][col] = 0

    larger_magnitude = (larger_magnitude / np.max(larger_magnitude) * 255).astype(np.uint8)
    return larger_magnitude


# double_thresholding 수행 high threshold value는 내장함수(otsu방식 이용)를 사용하여 구하고 low threshold값은 (high threshold * 0.4)로 구한다

def double_thresholding(src):
    ############################################
    # TODO                                     #
    # double_thresholding 완성                 #
    # dst     : 진짜 edge만 남은 image         #
    ###########################################
    (h, w) = src.shape
    strong = np.zeros((h, w), dtype=np.float)

    high_threshold_value, _ = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)
    low_threshold_value = high_threshold_value * 0.4
    for row in range(w):  # 스트롱 남기기
        for col in range(h):
            if high_threshold_value < src[row][col]:
                strong[row][col] = 1
            elif low_threshold_value > src[row][col]:
                strong[row][col] = 0

    for row in range(w):
        for col in range(h):
            check = 0
            x = row
            y = col
            dst = np.zeros((h, w), dtype=np.float)
            while (high_threshold_value > src[x][y] > low_threshold_value) and (not dst[x][y]):
                if 1 == (strong[x][y + 1] or strong[x + 1][y + 1] or strong[x + 1][y] or strong[x + 1][y - 1] or
                         strong[x][y - 1] or strong[x - 1][y - 1] or strong[x - 1][y] or strong[x - 1][y + 1]):
                    check = 1
                dst[x][y] = 1
                if high_threshold_value > src[x + 1][y] > low_threshold_value and (not dst[x + 1][y]):
                    x = x + 1
                    continue
                if high_threshold_value > src[x][y + 1] > low_threshold_value and (not dst[x][y + 1]):
                    y = y + 1
                    continue
                if high_threshold_value > src[x - 1][y] > low_threshold_value and (not dst[x - 1][y]):
                    x = x - 1
                    continue
                if high_threshold_value > src[x][y - 1] > low_threshold_value and (not dst[x][y - 1]):
                    y = y - 1
                    continue
                if high_threshold_value > src[x + 1][y + 1] > low_threshold_value and (not dst[x + 1][y + 1]):
                    x = x + 1
                    y = y + 1
                    continue
                if high_threshold_value > src[x - 1][y + 1] > low_threshold_value and (not dst[x - 1][y + 1]):
                    x = x - 1
                    y = y + 1
                    continue
                if high_threshold_value > src[x - 1][y - 1] > low_threshold_value and (not dst[x - 1][y - 1]):
                    x = x - 1
                    y = y - 1
                    continue
                if high_threshold_value > src[x + 1][y - 1] > low_threshold_value and (not dst[x + 1][y - 1]):
                    x = x + 1
                    y = y - 1
                    continue

                if check == 1:
                    strong = strong + dst
                    break

    dst = strong

    return dst


def my_canny_edge_detection(src, fsize=5, sigma=1, pad_type='zero'):
    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering을 해도 되고, DoG를 이용해 한번에 해도 됨
    Ix, Iy = apply_lowNhigh_pass_filter(src, fsize, sigma, pad_type)

    # magnitude와 angle을 구함
    magnitude = calcMagnitude(Ix, Iy)
    angle = calcAngle(Ix, Iy)

    # non-maximum suppression 수행
    larger_magnitude = non_maximum_supression(magnitude, angle)

    # 진짜 edge만 남김
    dst = double_thresholding(larger_magnitude)

    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    dst = my_canny_edge_detection(src)

    cv2.imshow('original', src)
    cv2.imshow('my canny edge detection', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
