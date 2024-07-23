import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

# 添加

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, out_cha=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, out_cha, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class UpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128):
        super(UpdateBlock, self).__init__()
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 16 * 16 * 9, 1, padding=0))

    def forward(self, imgf, coords1):
        imgf = imgf.permute(0,3,1,2)
        mask = .25 * self.mask(imgf)  # 缩放掩码以平衡梯度
        dflow = self.flow_head(imgf)
        coords1 = coords1 + dflow
        return mask, coords1

# 辅助函数

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

# 归一化
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma

# 前馈网络
def FeedForward(dim, hidden_dim, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )

# 注意力机制
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.norm = LayerNorm(dim)
        self.q_norm = RMSNorm(heads, dim_head)
        self.k_norm = RMSNorm(heads, dim_head)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, mask = None, attn_mask = None):
        x = self.norm(x)
        kv_input = default(context, x)
        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        q = self.q_norm(q)
        k = self.k_norm(k)
        dots = torch.matmul(q, k.transpose(-1, -2))
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)
        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Transformer 块
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
        self.norm = LayerNorm(dim)

    def forward(self, x, mask = None, attn_mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask, attn_mask = attn_mask) + x
            x = ff(x) + x
        return self.norm(x)

class NvDoc(nn.Module):
    def __init__(
        self, 
        *, 
        image_size, 
        patch_size, 
        num_classes, 
        dim, 
        depth, 
        heads, 
        mlp_dim, 
        channels = 3, 
        dim_head = 64, 
        dropout = 0., 
        emb_dropout = 0., 
        token_dropout_prob = None
    ):
        super().__init__()
        image_height, image_width = pair(image_size)

        self.decoder_norm_ft = nn.LayerNorm(dim)
        self.decoder_pred_ft = nn.Linear(dim, patch_size ** 2 * 2, bias=True)  # 解码器到patch

        self.update_block = UpdateBlock(patch_size ** 2 * 2)

        # 计算patching相关的内容
        assert divisible_by(image_height, patch_size) and divisible_by(image_width, patch_size), 'Image dimensions must be divisible by the patch size.'

        patch_height_dim, patch_width_dim = (image_height // patch_size), (image_width // patch_size)
        patch_dim = channels * (patch_size ** 2)

        self.channels = channels
        self.patch_size = patch_size

        self.to_patch_embedding = nn.Sequential(
            LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim),
        )

        self.pos_embed_height = nn.Parameter(torch.randn(patch_height_dim, dim))
        self.pos_embed_width = nn.Parameter(torch.randn(patch_width_dim, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # 最终的注意力池化查询
        self.attn_pool_queries = nn.Parameter(torch.randn(dim))
        self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)

        # 输出到logits
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_classes, bias = False)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    # 添加
    def unpatchify(self, x, ph, pw):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = 16
        x = x.reshape(shape=(x.shape[0], ph, pw, p*p*2))
        return x

    def coords_grid(self, batch, ht, wd):
        coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
        coords = torch.stack(coords[::-1], dim=0).float()
        return coords[None].repeat(batch, 1, 1, 1)

    def initialize_flow(self, img):
        N, C, H, W = img.shape
        coodslar = self.coords_grid(N, H, W).to(img.device)
        coords0 = self.coords_grid(N, H // 16, W // 16).to(img.device)
        coords1 = self.coords_grid(N, H // 16, W // 16).to(img.device)
        return coodslar, coords0, coords1

    def upsample_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 16, 16, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(16 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 16 * H, 16 * W)

    def forward(self, image):
        p, c, device = self.patch_size, self.channels, self.device

        # 处理图像
        assert image.ndim == 4 and image.shape[1] == c
        image_dims = image.shape[-2:]
        assert all([divisible_by(dim, p) for dim in image_dims]), f'Height and width of images must be divisible by the patch size {p}'

        ph, pw = map(lambda dim: dim // p, image_dims)

        x = rearrange(image, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p)
        x = x.contiguous()  # 添加这一行

        # 转换为patch嵌入
        x = self.to_patch_embedding(x)

        # 因子化的2D绝对位置嵌入
        pos = torch.stack(torch.meshgrid((
            torch.arange(ph, device=device),
            torch.arange(pw, device=device)
        ), indexing='ij'), dim=-1)

        pos = rearrange(pos, 'h w c -> (h w) c')
        h_pos, w_pos = pos.unbind(dim=-1)
        x = x + self.pos_embed_height[h_pos] + self.pos_embed_width[w_pos]

        # 嵌入dropout
        x = self.dropout(x)

        # 注意力
        x = self.transformer(x)

        x = self.decoder_norm_ft(x)
        x = self.decoder_pred_ft(x)
        x = self.unpatchify(x, ph, pw)

        coodslar, coords0, coords1 = self.initialize_flow(image)
        coords1 = coords1.detach()
        mask, coords1 = self.update_block(x, coords1)
        flow_up = self.upsample_flow(coords1 - coords0, mask)
        bm_up = coodslar + flow_up

        return bm_up