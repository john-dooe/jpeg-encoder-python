import cv2
import numpy as np

basic_quan_table_lum = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                 [12, 12, 14, 19, 26, 58, 60, 55],
                                 [14, 13, 16, 24, 40, 57, 69, 56],
                                 [14, 17, 22, 29, 51, 87, 80, 62],
                                 [18, 22, 37, 56, 68, 109, 103, 77],
                                 [24, 35, 55, 64, 81, 104, 113, 92],
                                 [49, 64, 78, 87, 103, 121, 120, 101],
                                 [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.uint8)

basic_quan_table_chroma = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                    [18, 21, 26, 66, 99, 99, 99, 99],
                                    [24, 26, 56, 99, 99, 99, 99, 99],
                                    [47, 66, 99, 99, 99, 99, 99, 99],
                                    [99, 99, 99, 99, 99, 99, 99, 99],
                                    [99, 99, 99, 99, 99, 99, 99, 99],
                                    [99, 99, 99, 99, 99, 99, 99, 99],
                                    [99, 99, 99, 99, 99, 99, 99, 99]], dtype=np.uint8)


def setup_quan_table(basic_quan_table, quality):
    if quality >= 50:
        quality = 200 - 2 * quality
    else:
        quality = 5000 / quality

    quan_table = (basic_quan_table * quality + 50) / 100
    quan_table = np.clip(quan_table, 1, 255)
    quan_table = quan_table.astype(np.uint8)
    return quan_table


def img_to_blocks(img, block_shape):
    height, width = img.shape[:2]
    block_height, block_width = block_shape
    shape = (height // block_height, width // block_width, block_height, block_width)
    strides = img.itemsize * np.array([width * block_height, block_width, width, 1])
    img_blocks = np.lib.stride_tricks.as_strided(img, shape, strides).astype('float64')
    img_blocks = np.reshape(img_blocks, (shape[0] * shape[1], block_height, block_width))
    return img_blocks


def blocks_to_img(img_blocks, img_shape):
    height, width = img_shape[:2]
    block_height, block_width = img_blocks.shape[-2:]
    shape = (height // block_height, width // block_width, block_height, block_width)
    img_blocks = np.reshape(img_blocks, shape)

    lines = []
    for line in img_blocks:
        lines.append(np.concatenate(line, axis=1))
    img = np.concatenate(lines, axis=0)

    return img


def block_preprocess(img_blocks, block_sum, quan_table):
    last_dc = 0
    dc_size_list = []
    dc_vli_list = []
    ac_first_byte_list = []
    ac_huffman_list = []
    ac_vli_list = []
    for i in range(block_sum):
        # 减去128，使取值在[-128, 127)
        block = img_blocks[i] - 128
        # DCT变换
        block_dct = cv2.dct(block)
        # 量化
        block_dct_quantized = np.round(block_dct / quan_table).astype(np.int32)
        # zig-zag取值
        block_dct_zig_zag = zig_zag(block_dct_quantized)
        # 左上角为dc，其余为ac
        dc = block_dct_zig_zag[0]
        ac = block_dct_zig_zag[1:]

        # 对dc进行delta编码
        dc_size, dc_vli = delta_encode(dc, last_dc)
        # 对ac进行rle编码
        ac_first_byte_block_list, ac_vli_block_list = run_length_encode(ac)

        dc_size_list.append(dc_size)
        dc_vli_list.append(dc_vli)
        ac_first_byte_list.append(ac_first_byte_block_list)
        ac_huffman_list += ac_first_byte_block_list
        ac_vli_list.append(ac_vli_block_list)

        last_dc = dc

    return dc_size_list, dc_vli_list, ac_first_byte_list, ac_huffman_list, ac_vli_list


def zig_zag(matrix):
    rows, columns = matrix.shape[:2]

    matrix_zig_zag = np.zeros(rows * columns, dtype=matrix.dtype)
    solution = [[] for _ in range(rows + columns - 1)]

    for i in range(rows):
        for j in range(columns):
            sum = i + j

            if sum % 2 == 0:
                solution[sum].insert(0, matrix[i][j])
            else:
                solution[sum].append(matrix[i][j])

    count = 0
    for i in solution:
        for j in i:
            matrix_zig_zag[count] = j
            count += 1

    return matrix_zig_zag


def run_length_encode(array):
    last_nonzero_index = 0
    for i, num in enumerate(array[::-1]):
        if num != 0:
            last_nonzero_index = len(array) - i
            break

    # (连续的0的个数, 下一个非0系数需要的位数, AC系数)
    # （4比特， 4比特， n比特）
    run_length = 0
    first_byte_list = []
    vli_list = []
    for i, num in enumerate(array):
        if i >= last_nonzero_index:
            first_byte_list.append(0)
            vli_list.append('')
            break
        elif num == 0 and run_length < 15:
            run_length += 1
        else:
            num_bits = variable_length_int_encode(num)
            size = len(num_bits)

            first_byte = int(bin(run_length)[2:].zfill(4) + bin(size)[2:].zfill(4), 2)

            first_byte_list.append(first_byte)
            vli_list.append(num_bits)
            run_length = 0

    return first_byte_list, vli_list


def delta_encode(dc, last_dc):
    num_bits = variable_length_int_encode(dc - last_dc)
    size = len(num_bits)

    return size, num_bits


def variable_length_int_encode(num):
    if num == 0:
        return ''
    elif num > 0:
        return bin(int(num))[2:]
    elif num < 0:
        bits = bin(abs(int(num)))[2:]
        return ''.join(map(lambda c: '0' if c == '1' else '1', bits))
