import cv2
import numpy as np


def my_filtering(src, ftype, fsize):
    (h, w) = src.shape
    dst = np.zeros((h, w))
    mask = np.ones(fsize, dtype=np.float)
    (mask_h, mask_w) = mask.shape
    mask_hight = int(mask_h / 2)        #마스크의 중앙지점이 되는 위치를 변수에 저장한다.
    mask_weight = int(mask_w / 2)

    if ftype == 'average':
        print('average filtering')
        ###################################################
        # TODO                                            #
        # mask 완성                                       #
        ###################################################
        mask = (1 / (mask_h * mask_w)) * mask       #average filter를 하는 모든 값이 1인 mask로 변경해준다.
        # mask 확인
        print(mask)

    elif ftype == 'sharpening':
        print('sharpening filtering')
        ##################################################
        # TODO                                           #
        # mask 완성                                      #
        ##################################################

        center = np.zeros((mask_h, mask_w), dtype=np.float)     #sharpening을 위해 배열의 중간이 1인 배열을 mask를 생성한다.
        center[mask_hight][mask_weight] = 1
        mask = 2 * center - (1 / (mask_h * mask_w)) * mask      #이론에서 배운 식에 맞게 마스크를 설정한다.
        # mask 확인
        print(mask)

    #########################################################
    # TODO                                                  #
    # dst 완성                                              #
    # dst : filtering 결과 image                            #
    #########################################################
    i = 0
    for row in range(h):
        for col in range(w):

            pix = 0
            if row - mask_hight >= 0 and row + mask_hight < w \
                    and col - mask_weight >= 0 and col + mask_weight < h:   #마스크의 범위가 src를 넘지않게 한다.

                for r in np.arange(mask_h):             #mask의 범위만큼 대응되는 src를 필터링해준다.
                    for c in np.arange(mask_w):
                        pix = pix + src[row - mask_hight + r][col - mask_weight + c] * mask[r][c]

            dst[row][col] = pix
            if pix >= 255:              #필셀값이 오버플로우가 발생할 경우의 예외를 처리해준다.
                dst[row][col] = 255
            elif pix <= 0:
                dst[row][col] = 0

    dst = (dst + 0.5).astype(np.uint8)

    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    # 3x3 filter
    dst_average = my_filtering(src, 'average', (3, 3))
    dst_sharpening = my_filtering(src, 'sharpening', (3, 3))

    # 원하는 크기로 설정
    # dst_average = my_filtering(src, 'average', (5,5))
    # dst_sharpening = my_filtering(src, 'sharpening', (5,5))

    # 11x13 filter
    # dst_average = my_filtering(src, 'average', (11, 13))
    # dst_sharpening = my_filtering(src, 'sharpening', (11, 13))

    cv2.imshow('original', src)
    cv2.imshow('average filter', dst_average)
    cv2.imshow('sharpening filter', dst_sharpening)
    cv2.waitKey()
    cv2.destroyAllWindows()
