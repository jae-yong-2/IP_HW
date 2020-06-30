import numpy as np
import cv2


def get_DCT(f, n=8):
    F = np.zeros((n, n))
    for u in range(n):
        for v in range(n):
            val = 0
            for x in range(n):
                for y in range(n):
                    val = val + f[x, y] * np.cos(((2 * x + 1) * u * np.pi) / (2 * n)) * np.cos(
                        ((2 * y + 1) * v * np.pi) / (2 * n))
            if u == 0:
                C_left = np.sqrt(1 / n)
            elif u != 0:
                C_left = np.sqrt(2 / n)

            if v == 0:
                C_right = np.sqrt(1 / n)
            elif v != 0:
                C_right = np.sqrt(2 / n)
            F[u, v] = C_left * C_right * val
    return F


def my_DCT(src, n=8):
    (h, w) = src.shape

    h_pad = h + (n - h % n)
    w_pad = w + (n - w % n)

    pad_img = np.zeros((h_pad, w_pad))
    pad_img[:h, :w] = src.copy()
    dst = np.zeros((h_pad, w_pad))

    for row_num in range(h_pad // n):
        for col_num in range(w_pad // n):
            dst[row_num * n: (row_num + 1) * n, col_num * n: (col_num + 1) * n] = get_DCT(
                pad_img[row_num * n: (row_num + 1) * n, col_num * n: (col_num + 1) * n], n)

    return dst[:h, :w]


def get_IDCT(F, n=8):
    f = np.zeros((n, n))
    for u in range(n):
        for v in range(n):
            val = 0
            for x in range(n):
                for y in range(n):
                    if x == 0:
                        C_left = np.sqrt(1 / n)
                    elif x != 0:
                        C_left = np.sqrt(2 / n)

                    if y == 0:
                        C_right = np.sqrt(1 / n)
                    elif y != 0:
                        C_right = np.sqrt(2 / n)
                    val = val + C_left * C_right * F[x, y] * np.cos(((2 * u + 1) * x * np.pi) / 16) * np.cos(
                        ((2 * v + 1) * y * np.pi) / 16)
                    f[u, v] = val
    return f


def my_IDCT(src, n=8):
    (h, w) = src.shape

    h_pad = h + (n - h % n)
    w_pad = w + (n - w % n)

    pad_img = np.zeros((h_pad, w_pad))
    pad_img[:h, :w] = src.copy()
    dst = np.zeros((h_pad, w_pad))

    for row in range(h_pad // n):
        for col in range(w_pad // n):
            dst[row * n: (row + 1) * n, col * n: (col + 1) * n] = get_IDCT(
                pad_img[row * n: (row + 1) * n, col * n: (col + 1) * n], n)

    return dst[:h, :w]


def Quantization_Luminance():
    luminance = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])
    return luminance


def my_JPEG_encoding(src, block_size=8):
    #####################################################
    # TODO                                              #
    # my_block_encoding 완성                            #
    # 입력변수는 알아서 설정(단, block_size는 8로 설정)   #
    # return                                            #
    # zigzag_value : encoding 결과(zigzag까지)          #
    #####################################################
    dst = src.copy() - 128
    dst = my_DCT(dst, block_size)
    # dst = src
    zigzag_value = np.zeros(dst.size)
    h, w = dst.shape

    i = 0
    for row in range(h // block_size):
        for col in range(w // block_size):

            dst[row * block_size:(row + 1) * block_size, col * block_size:(col + 1) * block_size] = np.round(
                dst[row * block_size:(row + 1) * block_size,
                col * block_size:(col + 1) * block_size] / Quantization_Luminance())

            r = row * block_size
            c = col * block_size

            r_zero = r
            c_zero = c

            zigzag_value[i] = dst[r][c]
            i = i + 1
            count = 0
            while 1:  # block

                if count == 63:
                    break
                if r == r_zero and not (c == (c_zero + block_size - 1)):  # side

                    c = c + 1
                    zigzag_value[i] = dst[r][c]
                    i = i + 1
                    count = count + 1
                elif c == c_zero and not (r == (r_zero + block_size - 1)):  # side

                    r = r + 1
                    zigzag_value[i] = dst[r][c]
                    i = i + 1
                    count = count + 1

                elif r == (r_zero + block_size - 1):

                    c = c + 1
                    zigzag_value[i] = dst[r][c]
                    i = i + 1
                    count = count + 1

                elif c == (c_zero + block_size - 1):

                    r = r + 1
                    zigzag_value[i] = dst[r][c]
                    i = i + 1
                    count = count + 1

                if count == 63:
                    break

                if r == r_zero or c == (c_zero + block_size - 1):  # down

                    while c != c_zero and r != r_zero + block_size - 1:
                        r = r + 1
                        c = c - 1
                        zigzag_value[i] = dst[r][c]
                        i = i + 1
                        count = count + 1

                elif c == c_zero or r == (r_zero + block_size - 1):  # up

                    while r != r_zero and c != c_zero + block_size - 1:
                        r = r - 1
                        c = c + 1
                        zigzag_value[i] = dst[r][c]
                        i = i + 1
                        count = count + 1

    zigzag_value = zigzag_value[::-1]
    for x in range(int(zigzag_value.size / (block_size * block_size))):
        for y in range(block_size * block_size):

            if zigzag_value[x * block_size * block_size + y] != 0:
                break

            else:
                zigzag_value[x * block_size * block_size + y] = np.nan

    zigzag_value = zigzag_value[::-1]
    return zigzag_value


def my_JPEG_decoding(zigzag_value, block_size=8):
    #####################################################
    # TODO                                              #
    # my_JPEG_decoding 완성                             #
    # 입력변수는 알아서 설정(단, block_size는 8로 설정)   #
    # return                                            #
    # dst : decoding 결과 이미지                         #
    #####################################################
    for i in range(len(zigzag_value)):

        if np.isnan(zigzag_value[i]):
            zigzag_value[i] = 0

    dst = np.zeros((int(zigzag_value.size ** 0.5), int(zigzag_value.size ** 0.5)))

    h, w = dst.shape

    i = 0
    for row in range(h // block_size):
        for col in range(w // block_size):

            r = row * block_size
            c = col * block_size

            r_zero = r
            c_zero = c

            dst[r][c] = zigzag_value[i]
            i = i + 1
            count = 0
            while 1:  # block

                if count == 63:
                    break
                if r == r_zero and not (c == (c_zero + block_size - 1)):  # side

                    c = c + 1
                    dst[r][c] = zigzag_value[i]
                    i = i + 1
                    count = count + 1
                elif c == c_zero and not (r == (r_zero + block_size - 1)):  # side

                    r = r + 1
                    dst[r][c] = zigzag_value[i]
                    i = i + 1
                    count = count + 1

                elif r == (r_zero + block_size - 1):

                    c = c + 1
                    dst[r][c] = zigzag_value[i]
                    i = i + 1
                    count = count + 1

                elif c == (c_zero + block_size - 1):

                    r = r + 1
                    dst[r][c] = zigzag_value[i]
                    i = i + 1
                    count = count + 1

                if count == 63:
                    break

                if r == r_zero or c == (c_zero + block_size - 1):  # down

                    while c != c_zero and r != r_zero + block_size - 1:
                        r = r + 1
                        c = c - 1
                        dst[r][c] = zigzag_value[i]
                        i = i + 1
                        count = count + 1

                elif c == c_zero or r == (r_zero + block_size - 1):  # up

                    while r != r_zero and c != c_zero + block_size - 1:
                        r = r - 1
                        c = c + 1
                        dst[r][c] = zigzag_value[i]
                        i = i + 1
                        count = count + 1

    for row in range(h // block_size):
        for col in range(w // block_size):
            dst[row * block_size:(row + 1) * block_size, col * block_size:(col + 1) * block_size] = np.round(
                dst[row * block_size:(row + 1) * block_size,
                col * block_size:(col + 1) * block_size] * Quantization_Luminance())

    dst = my_IDCT(dst, block_size)
    dst = dst + 128
    dst = dst.astype(np.uint8)

    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    """ 
    #이론ppt에 나와있는 배열 테스트하실분만 해보시면 될 것 같습니다.
    #참고로 이론 ppt에 나온 값하고 다르게 나옵니다.
    #(예를 들어 인덱스가 [5,5]인 68을 예로 들면 68 - 128을 하면 -60이 나와야 하는데
    #이론ppt에는 -65로 값이 잘못나와있습니다. 뭔가 값이 조금씩 다르게 나옵니다. 그러니 참고용으로만 사용해주세요)
    
    """
    # src = np.array(
    #     [[52, 55, 61, 66, 70, 61, 64, 73],
    #      [63, 59, 66, 90, 109, 85, 69, 72],
    #      [62, 59, 68, 113, 144, 104, 66, 73],
    #      [63, 58, 71, 122, 154, 106, 70, 69],
    #      [67, 61, 68, 104, 126, 88, 68, 70],
    #      [79, 65, 60, 70, 77, 68, 58, 75],
    #      [85, 71, 64, 59, 55, 61, 65, 83],
    #      [87, 79, 69, 68, 65, 76, 78, 94]])
    src = src.astype(np.float)
    zigzag_value = my_JPEG_encoding(src)
    print(zigzag_value[:10])
    dst = my_JPEG_decoding(zigzag_value)
    src = src.astype(np.uint8)
    cv2.imshow('original', src)
    cv2.imshow('result', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
