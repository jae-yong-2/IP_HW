import numpy as np
import cv2
import matplotlib.pyplot as plt


def my_calcHist(src):  # 히스토그램을 구하는 함수
    ###############################
    # TODO                        #
    # my_calcHist완성             #
    # src : input image           #
    # hist : src의 히스토그램      #
    ###############################
    (h, w) = src.shape  # src의 높이와 넓이를 구한다.
    hist = np.zeros((256,), dtype=np.int)  # 반환할 hist의 이차원 배열 생성

    for row in range(h):  # src의 필셀값의 횟수를 hist에 저장
        for col in range(w):
            intensity = src[row, col]
            hist[intensity] = hist[intensity] + 1

    return hist  # 히스토그랭 반환


def my_normalize_hist(hist, pixel_num):  # 히스토그램을 전체 픽셀수로 나누는 함수
    ########################################################
    # TODO                                                 #
    # my_normalize_hist완성                                #
    # hist : 히스토그램                                     #
    # pixel_num : image의 전체 픽셀 수                      #
    # normalized_hist : 히스토그램값을 총 픽셀수로 나눔       #
    ########################################################
    normalized_hist = np.zeros((256,), dtype=np.float)  # 반환할 배열 생성
    for i in range(256):  # 히스토그램을 다돌면서 전체 픽셀로 나눔
        normalized_hist[i] = hist[i] / pixel_num

    return normalized_hist  # 값을 반환


def my_PDF2CDF(pdf):  # normalized_hist를 누적시킨다.
    ########################################################
    # TODO                                                 #
    # my_PDF2CDF완성                                       #
    # pdf : normalized_hist                                #
    # cdf : pdf의 누적                                     #
    ########################################################
    cdf = np.zeros((256,), dtype=np.float)
    cdf[0] = pdf[0]  # 첫값을 설정
    for i in range(255):  # 반복문을 통해 값을 누적
        cdf[i + 1] = cdf[i] + pdf[i + 1]

    return cdf  # 값을 반환


def my_denormalize(normalized, gray_level):  # 평활화를 위해서
    ######################################################## 누적된 pdf값을 gray_level로 곱하여
    # TODO                                                 # denoralized를 얻어낸다
    # my_denormalize완성                                   #
    # normalized : 누적된pdf값(cdf)                        #
    # gray_level : max_gray_level                          #
    # denormalized : normalized와 gray_level을 곱함        #
    ########################################################
    denormalized = np.zeros((256,), dtype=np.float)
    for i in range(len(normalized)):
        denormalized[i] = normalized[i] * gray_level  # 평활화를 위에 gray_level을 곱함

    return denormalized


def my_calcHist_equalization(denormalized, hist):  # 히스토그램을 평활화한다.
    ###################################################################
    # TODO                                                            #
    # my_calcHist_equalization완성                                    #
    # denormalized : output gray_level(정수값으로 변경된 gray_level)   #
    # hist : 히스토그램                                                #
    # hist_equal : equalization된 히스토그램                           #
    ####################################################################
    hist_equal = np.zeros(hist.shape, dtype=np.uint32)

    for i in range(len(hist)):  # 앞에서 구한 denormalized함수를 이용해
        hist_equal[denormalized[i]] = hist[i]  # hist의 값을 평활화한다.

    return hist_equal  # 평활화한 히스토그램을 반환


def my_equal_img(src, output_gray_level):
    ###################################################################
    # TODO                                                            #
    # my_equal_img완성                                                #
    # src : input image                                               #
    # output_gray_level : denormalized(정수값으로 변경된 gray_level)   #
    # dst : equalization된 결과 이미지                                 #
    ###################################################################

    dst= np.zeros(src.shape, dtype=np.uint8)
    (h, w) = src.shape

    for row in range(h):
        for col in range(w):
            dst[row][col] = output_gray_level[src[row][col]]
    #src 는 0~81까지의 밝기를 가지고있다. 여기서 denormalized한 값은 인덱스 0~81까지 그에 맞는
    #평준화된 값을 각각의 장소에 저장하고 있다. 따라서 src에 저장된 밝기값을 output_gray_level에
    #넣으면 평준화된 값이 나오고 값을 dst에 저장한다.

    return dst


# input_image의  equalization된 histogram & image 를 return
def my_hist_equal(src):
    (h, w) = src.shape
    max_gray_level = 255
    histogram = my_calcHist(src)
    normalized_histogram = my_normalize_hist(histogram, h * w)
    normalized_output = my_PDF2CDF(normalized_histogram)
    denormalized_output = my_denormalize(normalized_output, max_gray_level)
    output_gray_level = denormalized_output.astype(int)
    hist_equal = my_calcHist_equalization(output_gray_level, histogram)

    ### dst : equalization 결과 image
    dst = my_equal_img(src, output_gray_level)

    return dst, hist_equal


if __name__ == '__main__':
    src = cv2.imread('fruits_div3.jpg', cv2.IMREAD_GRAYSCALE)
    hist = my_calcHist(src)
    dst, hist_equal = my_hist_equal(src)

    cv2.imshow('original', src)
    binX = np.arange(len(hist))
    plt.title('my histogram')
    plt.bar(binX, hist, width=0.5, color='g')
    plt.show()

    cv2.imshow('equalizetion after image', dst)
    binX = np.arange(len(hist_equal))
    plt.title('my histogram equalization')
    plt.bar(binX, hist_equal, width=0.5, color='g')
    plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()
