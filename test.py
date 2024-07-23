#encoding=utf-8
import torch
from models import NvDoc

n = NvDoc(
    image_size = 256,
    patch_size = 16,
    num_classes = 1000,
    dim = 768,
    heads = 8,
    mlp_dim=1024,
    dropout=0.1,
    emb_dropout=0.1,
    token_dropout_prob=0,
    depth=8
)

# 创建一个单独的图像张量
image = torch.randn(1, 3, 16, 32)  # 批次大小为1，3个通道，256x256像素

preds = n(image)

print(preds.shape)