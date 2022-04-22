import struct
import numpy as np


# Start of Image
def write_soi():
    marker = b'\xff\xd8'
    return marker


# Application0
def write_app0():
    marker = b'\xff\xe0'
    app0_type = b'JFIF\x00'
    version = b'\x01\x01'
    units = b'\x00'
    density = b'\x00\x01\x00\x01'
    thumbnail = b'\x00\x00'

    app0_data = app0_type + version + units + density + thumbnail
    size = struct.pack('>H', len(app0_data) + 2)

    app0 = marker + size + app0_data
    return app0


# Define Quantization Table
def write_dqt(quan_table, num):
    marker = b'\xff\xdb'
    # 默认顺序为：亮度、色度
    if num == 0:
        quan_table_info = b'\x00'
    else:
        quan_table_info = b'\x01'
    quan_table_bytes = quan_table.tobytes()

    dqt_data = quan_table_info + quan_table_bytes
    size = struct.pack('>H', len(dqt_data) + 2)

    dqt = marker + size + dqt_data
    return dqt


# Start of Frame
def write_sof(height, width, channel_num):
    marker = b'\xff\xc0'
    # 默认8-bit色深
    precision = b'\x08'
    y_image = struct.pack('>H', height)
    x_image = struct.pack('>H', width)
    # 通道数
    components_num = struct.pack('>B', channel_num)

    components = b''
    for i in range(channel_num):
        component_index = struct.pack('>B', i + 1)
        # 默认YCbCr 4:4:4 采样
        sample_factor = b'\x11'

        # 默认第一个量化表负责亮度，第二个负责色度
        if i == 0:
            quan_table_index = b'\x00'
        else:
            quan_table_index = b'\x01'

        components += component_index + sample_factor + quan_table_index

    sof_data = precision + y_image + x_image + components_num + components
    size = struct.pack('>H', len(sof_data) + 2)
    sof = marker + size + sof_data
    return sof


# Define Huffman Table
def write_dht(code_dict, num):
    marker = b'\xff\xc4'
    # 默认顺序为：亮度dc、亮度ac、色度dc、色度ac
    if num == 0:
        huffman_table_info = b'\x00'
    elif num == 1:
        huffman_table_info = b'\x10'
    elif num == 2:
        huffman_table_info = b'\x01'
    else:
        huffman_table_info = b'\x11'

    count = 0
    length_array = np.zeros(16, dtype=np.uint8)
    symbol_array = np.zeros(len(code_dict), dtype=np.uint8)
    for symbol, code in code_dict.items():
        length_array[len(code) - 1] += 1
        symbol_array[count] = symbol
        count += 1

    length_bytes = length_array.tobytes()
    symbol_bytes = symbol_array.tobytes()

    dht_data = huffman_table_info + length_bytes + symbol_bytes
    size = struct.pack('>H', len(dht_data) + 2)
    dht = marker + size + dht_data

    return dht


# Start of Scan
def write_sos(channel_num, image_data):
    marker = b'\xff\xda'
    # 通道数
    components_num = struct.pack('>B', channel_num)

    components = b''
    for i in range(channel_num):
        component_index = struct.pack('>B', i + 1)

        # 默认第一个哈夫曼表负责亮度，第二个负责色度
        if i == 0:
            huffman_table_index = b'\x00'
        else:
            huffman_table_index = b'\x11'

        components += component_index + huffman_table_index

    end = b'\x00\x3f\x00'

    sos_data = components_num + components + end
    size = struct.pack('>H', len(sos_data) + 2)
    sos = marker + size + sos_data + image_data
    return sos


# End of Image
def write_eoi():
    marker = b'\xff\xd9'
    return marker


def write_jpeg(file_name, height, width, channel_num, image_data, quan_table_list, huffman_code_dict_list):
    soi = write_soi()
    app0 = write_app0()

    # 多个量化表
    count = 0
    dqt = b''
    for quan_table in quan_table_list:
        dqt += write_dqt(quan_table, count)
        count += 1

    sof = write_sof(height, width, channel_num)

    # 多个哈夫曼表
    count = 0
    dht = b''
    for code_dict in huffman_code_dict_list:
        dht += write_dht(code_dict, count)
        count += 1

    sos = write_sos(channel_num, image_data)
    eoi = write_eoi()

    jpeg = soi + app0 + dqt + sof + dht + sos + eoi
    f = open(file_name, 'wb')
    f.write(jpeg)
    f.close()
