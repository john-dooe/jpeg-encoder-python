# jpeg-encoder-python

用 Python 实现的简单 JPEG 编码器

目前仅支持 Baseline DCT、哈夫曼编码、YUV 4:4:4

## 用法

```bash
python3 jpeg_encoder.py [原图像] -o [JPEG编码后的文件名]
```

加上 -l 或者 --lossless 可以使用无损压缩，否则默认使用有损压缩
