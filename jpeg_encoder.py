import cv2
import argparse
import numpy as np

from encoder.color_encoder import color_encoder
from encoder.grayscale_encoder import grayscale_encoder


# 判断图片是不是灰度图
def is_grayscale(img):
    img_b, img_g, img_r = cv2.split(img)
    if np.array_equal(img_b, img_g) and np.array_equal(img_g, img_r):
        return True
    else:
        return False


# 如果图片宽高不能被8整除，加上黑色边框
def fill(img, height, width, channel_num):
    img_filled = img.copy()

    if height % 8 != 0:
        filler = np.ones((8 - (height % 8), width, channel_num), dtype=np.uint8) * 128
        img_filled = np.concatenate([img, filler], axis=0)

    if width % 8 != 0:
        filler = np.ones((height + 8 - (height % 8), 8 - (width % 8), channel_num), dtype=np.uint8) * 128
        img_filled = np.concatenate([img_filled, filler], axis=1)

    return img_filled


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', help='The path of the image you need to encode')
    parser.add_argument('-o', '--out', default='jpeg.jpg', help='The filename of the JPEG encoded image')
    parser.add_argument('-q', '--quality', default=75, type=int, choices=range(1, 101),
                        help='JPEG compression quality (1-100)')
    args = parser.parse_args()

    img_path = args.img_path
    saved_jpeg_name = args.out
    quality = args.quality

    if not saved_jpeg_name.endswith('.jpg'):
        saved_jpeg_name += '.jpg'

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    height, width = img.shape[:2]

    # 灰度图
    if is_grayscale(img):
        print(f'{img_path} is a grayscale image.')
        img = fill(img, height, width, 1)
        img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayscale_encoder(saved_jpeg_name, img_grayscale, height, width, quality)

    # RGB彩色图
    else:
        print(f'{img_path} is a color image.')
        img = fill(img, height, width, 3)
        color_encoder(saved_jpeg_name, img, height, width, quality)
