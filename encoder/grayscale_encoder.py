from utils.utils import *
from utils.file_writer import *
from utils.huffman import HuffmanEncoder


def grayscale_encoder(file_name, img, real_height, real_width, is_lossless):
    block_shape = (8, 8)

    filled_height, filled_width = img.shape[:2]
    block_sum = filled_height // block_shape[0] * filled_width // block_shape[1]

    img_blocks = img_to_blocks(img, block_shape)

    if is_lossless:
        dc_size_list, dc_vli_list, ac_first_byte_list, ac_huffman_list, ac_vli_list = block_preprocess(
            img_blocks, block_sum, quan_table_lossless)
    else:
        dc_size_list, dc_vli_list, ac_first_byte_list, ac_huffman_list, ac_vli_list = block_preprocess(
            img_blocks, block_sum, quan_table_lum)

    # 构造哈夫曼树
    huffman_encoder_dc = HuffmanEncoder(dc_size_list)
    code_dict_dc = huffman_encoder_dc.code_dict
    huffman_encoder_ac = HuffmanEncoder(ac_huffman_list)
    code_dict_ac = huffman_encoder_ac.code_dict

    # 对dc系数的大小进行哈夫曼编码
    dc_size_list_encoded = huffman_encoder_dc.encode(dc_size_list)

    image_data_bits = ''
    for i in range(block_sum):
        # 对ac系数的0的数量及大小进行哈夫曼编码
        ac_first_byte_encoded = huffman_encoder_ac.encode(ac_first_byte_list[i])

        block_encoded = dc_size_list_encoded[i] + dc_vli_list[i]
        for j in range(len(ac_first_byte_encoded)):
            block_encoded += ac_first_byte_encoded[j] + ac_vli_list[i][j]

        image_data_bits += block_encoded

    # 补1
    if len(image_data_bits) % 8 != 0:
        image_data_bits += (8 - (len(image_data_bits) % 8)) * '1'

    image_data = int(image_data_bits, 2).to_bytes(len(image_data_bits) // 8, 'big')

    # FF替换为FF00
    image_data = image_data.replace(b'\xff', b'\xff\x00')

    if is_lossless:
        write_jpeg(file_name, real_height, real_width, 1, image_data, [quan_table_lossless],
                   [code_dict_dc, code_dict_ac])
    else:
        write_jpeg(file_name, real_height, real_width, 1, image_data, [quan_table_lum], [code_dict_dc, code_dict_ac])
