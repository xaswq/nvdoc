#encoding=utf-8
import torch
from navit.main import NaViT


n = NaViT(
    image_size = 256,
    patch_size = 16,
    num_classes = 1000,
    dim = 1024,
    heads = 16,
    mlp_dim=1024,
    dropout=0.1,
    emb_dropout=0.1,
    token_dropout_prob=0,
    depth=12  # 加上这个，不然无法运行

)



images = [
     [torch.randn(3, 32, 48)],

]


preds = n(images)

print(preds.shape)
