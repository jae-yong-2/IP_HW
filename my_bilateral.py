import numpy as np
import cv2
import my_padding as my_p


def my_get_Gaussian2D_mask(msize, sigma=1):
    y, x = np.mgrid[-(msize // 2):(msize // 2) + 1, -(msize // 2):(msize // 2) + 1]
    # 2차 gaussian mask 생성
    gaus2D = 1 / (2 * np.pi * sigma ** 2) * np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    # mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)
    print(gaus2D)
    return gaus2D


def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape
    (m_h, m_w) = mask.shape
    pad_img = my_p.my_padding(src, (m_h // 2, m_w // 2), pad_type)
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + m_h, col:col + m_w] * mask)
    return dst


def my_normalize(src):
    dst = src.copy()
    dst = dst - np.min(dst)
    dst = dst / np.max(dst) * 255
    return dst.astype(np.uint8)


def my_bilateral(src, msize, sigma, sigma_r, pad_type='zero'):
    ############################################
    # TODO                                     #
    # my_bilateral 함수 완성                   #
    # src : 원본 image                         #
    # msize : mask size                        #
    # sigma : sigma_x, sigma_y 값              #
    # sigma_r : sigma_r값                      #
    # pad_type : padding type                  #
    # dst : bilateral filtering 결과 image     #
    ############################################
    h, w = src.shape
    dst = np.zeros(src.shape, dtype=np.float)
    # mask
    gaus2D_mask = my_get_Gaussian2D_mask(msize, sigma)
    mask = np.zeros((msize, msize), dtype=np.float)
    m_s = msize // 2

    pad_img = my_p.my_padding(src, (msize // 2, msize // 2), 'repetition')
    for i_x in range(m_s, h):
        for i_y in range(m_s, w):
            # 마스크
            for m_x in range(msize):
                for m_y in range(msize):
                    mask[m_x, m_y] = np.exp(
                        -((pad_img[i_x][i_y] - pad_img[i_x - m_s + m_x ][i_y - m_s + m_y-1]) ** 2) / (2 * (sigma_r ** 2)))

            my_bilateral_maks = gaus2D_mask * mask
            my_bilateral_maks = my_bilateral_maks / np.sum(my_bilateral_maks)
            if i_x == 53 and i_y == 123:  # 마스크 맞는지 체크부분
                print(my_bilateral_maks)
                mask_img = cv2.resize(my_bilateral_maks, (200, 200), interpolation=cv2.INTER_NEAREST)
                mask_img = my_normalize(mask_img)
                cv2.imshow('mask', mask_img)
                img = src[i_x - m_s-2: i_x + m_s-1, i_y - m_s-2: i_y + m_s-1]
                img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_NEAREST)
                img = my_normalize(img)
                cv2.imshow('mask img', img)
            pad_img[i_x][i_y] = np.sum(pad_img[i_x - msize // 2: i_x + msize // 2 + 1,
                                       i_y - msize // 2: i_y + msize // 2 + 1] * my_bilateral_maks)

    # mask의 총 합 = 1
    dst = pad_img[msize // 2:h, msize // 2:w] / 255

    return dst


if __name__ == '__main__':
    src = cv2.imread('Penguins_noise.png', cv2.IMREAD_GRAYSCALE)
    dst = my_bilateral(src, 5, 3, 40)

    gaus2D = my_get_Gaussian2D_mask(5, sigma=1)
    dst_gaus2D = my_filtering(src, gaus2D)
    dst_gaus2D = my_normalize(dst_gaus2D)

    cv2.imshow('original', src)
    cv2.imshow('my gaussian', dst_gaus2D)
    cv2.imshow('my bilateral', dst)

    cv2.waitKey()
    cv2.destroyAllWindows()
