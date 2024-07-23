import numpy as np
import cv2

def resize_keep_aspect_ratio(image, max_size=256):
    h, w = image.shape[:2]
    if h > w:
        new_h = min(h, max_size)
        new_w = int(w * (new_h / h))
    else:
        new_w = min(w, max_size)
        new_h = int(h * (new_w / w))
    
    # 确保宽和高都是16的倍数
    new_h = (new_h // 16) * 16
    new_w = (new_w // 16) * 16
    
    return cv2.resize(image, (new_w, new_h))

def test_resize(input_shape):
    # 创建随机图像
    image = np.random.rand(*input_shape).astype(np.float32)
    
    # 应用resize函数
    resized_image = resize_keep_aspect_ratio(image)
    
    # 打印结果
    print(f"Original shape: {image.shape}")
    print(f"Resized shape: {resized_image.shape}")
    print(f"Is height multiple of 16? {resized_image.shape[0] % 16 == 0}")
    print(f"Is width multiple of 16? {resized_image.shape[1] % 16 == 0}")
    print(f"Max dimension <= 256? {max(resized_image.shape[:2]) <= 256}")
    print("-----")

# 测试不同的输入尺寸
test_shapes = [
    (300, 200, 3),  # 高大于宽，都大于256
    (200, 300, 3),  # 宽大于高，都大于256
    (400, 100, 3),  # 高远大于宽
    (100, 400, 3),  # 宽远大于高
    (250, 250, 3),  # 正方形，接近256
    (100, 100, 3),  # 小正方形
    (500, 500, 3),  # 大正方形
]

for shape in test_shapes:
    test_resize(shape)