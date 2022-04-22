# jpeg-encoder-python

用 Python 实现的简单 JPEG 编码器

目前仅支持 Baseline DCT、哈夫曼编码、YUV 4:4:4

## 用法

```
python3 jpeg_encoder.py [原图像路径] -o [编码后的JPEG图片文件名] -q [编码的质量系数（1-100）]
```

`-o` 以及 `-q` 是可选项

如果不加 `-o`，默认文件名为 jpeg.jpg

如果不加 `-q`，默认质量系数为 75
