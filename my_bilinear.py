import cv2
import numpy as np


def my_bilinear(src, dst_shape):
    #########################
    # TODO                  #
    # my_bilinear 완성      #
    #########################
    (src_h, src_w) = src.shape
    (dst_h, dst_w) = dst_shape
    scale = dst_h / src_h
    dst = np.zeros(dst_shape, dtype="uint8")

    for row in range(dst_h):
        for col in range(dst_w):
            r = int(row / scale)                # dst의 row를 돌면서 그에 대응하는 src의 row값을 찾는다.
            c = int(col / scale)                # 위와 마찬 가지로 src의 col값을 찾는다.
            num1 = src[r][c]                    # 공식에서 사용되는      # f(r,c)의 값
            num2 = src[r][min(c + 1, src_w - 1)]                        # f(r, c+1)
            num3 = src[min(r + 1, src_h - 1)][c]                        # f(r+1, c)
            num4 = src[min(r + 1, src_h - 1)][min(c + 1, src_w - 1)]    # f(r+1, c+1)

            s = (col - scale * c) / scale
            t = (row - scale * r) / scale
            # 공식에서 사용되는 s와 t의 값 (사이 간격을 1이라 생각하고 사용)
            fix = int((1 - s) * (1 - t) * num1 + s * (1 - t) * num2 + (1 - s) * t * num3 + s * t * num4)
            # 선형간섭으로 dst[row, col]을 얻을 수 있는 식 (1-s)(1-t) f(r,c) + s(1-t) f(r,c+1) + (1-s)t f(r+1, c) + st f(r+1, c+1)
            dst[row, col] = fix

    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    # 이미지 크기 ??x??로 변경

    my_dst_mini = my_bilinear(src, (128, 128))

    # 이미지 크기 512x512로 변경(Lena.png 이미지의 shape는 (512, 512))
    my_dst = my_bilinear(my_dst_mini, (512, 512))
    print(my_dst)
    cv2.imshow('original', src)
    cv2.imshow('my bilinear mini', my_dst_mini)
    cv2.imshow('my bilinear', my_dst)

    cv2.waitKey()
    cv2.destroyAllWindows()
