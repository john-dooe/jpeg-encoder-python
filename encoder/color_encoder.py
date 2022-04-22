from utils.utils import *
from utils.file_writer import *
from utils.huffman import HuffmanEncoder


def color_encoder(file_name, img, real_height, real_width, is_lossless):
    block_shape = (8, 8)

    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img_y, img_cr, img_cb = cv2.split(img_ycrcb)

    filled_height, filled_width = img_y.shape[:2]
    block_sum = filled_height // block_shape[0] * filled_width // block_shape[1]

    img_y_blocks = img_to_blocks(img_y, block_shape)
    img_cr_blocks = img_to_blocks(img_cr, block_shape)
    img_cb_blocks = img_to_blocks(img_cb, block_shape)

    if is_lossless:
        dc_y_size_list, dc_y_vli_list, ac_y_first_byte_list, ac_y_huffman_list, ac_y_vli_list = block_preprocess(
            img_y_blocks,
            block_sum,
            quan_table_lossless)
        dc_cr_size_list, dc_cr_vli_list, ac_cr_first_byte_list, ac_cr_huffman_list, ac_cr_vli_list = block_preprocess(
            img_cr_blocks, block_sum, quan_table_lossless)
        dc_cb_size_list, dc_cb_vli_list, ac_cb_first_byte_list, ac_cb_huffman_list, ac_cb_vli_list = block_preprocess(
            img_cb_blocks, block_sum, quan_table_lossless)
    else:
        dc_y_size_list, dc_y_vli_list, ac_y_first_byte_list, ac_y_huffman_list, ac_y_vli_list = block_preprocess(
            img_y_blocks,
            block_sum,
            quan_table_lum)
        dc_cr_size_list, dc_cr_vli_list, ac_cr_first_byte_list, ac_cr_huffman_list, ac_cr_vli_list = block_preprocess(
            img_cr_blocks, block_sum, quan_table_chroma)
        dc_cb_size_list, dc_cb_vli_list, ac_cb_first_byte_list, ac_cb_huffman_list, ac_cb_vli_list = block_preprocess(
            img_cb_blocks, block_sum, quan_table_chroma)

    # 为亮度构造哈夫曼树
    huffman_encoder_dc_y = HuffmanEncoder(dc_y_size_list)
    code_dict_dc_y = huffman_encoder_dc_y.code_dict
    huffman_encoder_ac_y = HuffmanEncoder(ac_y_huffman_list)
    code_dict_ac_y = huffman_encoder_ac_y.code_dict

    # 为色度构造哈夫曼树
    huffman_encoder_dc_chroma = HuffmanEncoder(dc_cr_size_list + dc_cb_size_list)
    code_dict_dc_chroma = huffman_encoder_dc_chroma.code_dict
    huffman_encoder_ac_chroma = HuffmanEncoder(ac_cr_huffman_list + ac_cb_huffman_list)
    code_dict_ac_chroma = huffman_encoder_ac_chroma.code_dict

    # 对dc系数的大小进行哈夫曼编码
    dc_y_size_list_encoded = huffman_encoder_dc_y.encode(dc_y_size_list)
    dc_cr_size_list_encoded = huffman_encoder_dc_chroma.encode(dc_cr_size_list)
    dc_cb_size_list_encoded = huffman_encoder_dc_chroma.encode(dc_cb_size_list)

    image_data_bits = ''
    for i in range(block_sum):
        # 对ac系数的0的数量及大小进行哈夫曼编码
        ac_y_first_byte_encoded = huffman_encoder_ac_y.encode(ac_y_first_byte_list[i])
        ac_cr_first_byte_encoded = huffman_encoder_ac_chroma.encode(ac_cr_first_byte_list[i])
        ac_cb_first_byte_encoded = huffman_encoder_ac_chroma.encode(ac_cb_first_byte_list[i])

        # 写入亮度数据
        block_encoded = dc_y_size_list_encoded[i] + dc_y_vli_list[i]
        for j in range(len(ac_y_first_byte_encoded)):
            block_encoded += ac_y_first_byte_encoded[j] + ac_y_vli_list[i][j]

        # 写入cb数据
        block_encoded += dc_cb_size_list_encoded[i] + dc_cb_vli_list[i]
        for j in range(len(ac_cb_first_byte_encoded)):
            block_encoded += ac_cb_first_byte_encoded[j] + ac_cb_vli_list[i][j]

        # 写入cr数据
        block_encoded += dc_cr_size_list_encoded[i] + dc_cr_vli_list[i]
        for j in range(len(ac_cr_first_byte_encoded)):
            block_encoded += ac_cr_first_byte_encoded[j] + ac_cr_vli_list[i][j]

        image_data_bits += block_encoded

    # 补1
    if len(image_data_bits) % 8 != 0:
        image_data_bits += (8 - (len(image_data_bits) % 8)) * '1'

    image_data = int(image_data_bits, 2).to_bytes(len(image_data_bits) // 8, 'big')

    # FF替换为FF00
    image_data = image_data.replace(b'\xff', b'\xff\x00')

    if is_lossless:
        write_jpeg(file_name, real_height, real_width, 3, image_data, [quan_table_lossless, quan_table_lossless],
                   [code_dict_dc_y, code_dict_ac_y, code_dict_dc_chroma, code_dict_ac_chroma])
    else:
        write_jpeg(file_name, real_height, real_width, 3, image_data, [quan_table_lum, quan_table_chroma],
                   [code_dict_dc_y, code_dict_ac_y, code_dict_dc_chroma, code_dict_ac_chroma])
