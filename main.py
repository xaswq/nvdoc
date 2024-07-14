from functools import partial
from typing import List, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence


#add

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
            nn.Conv2d(256, 16 * 16 * 9, 1, padding=0)) #change to 16

    def forward(self, imgf, coords1):
        imgf = imgf.permute(0,3,1,2)
        mask = .25 * self.mask(imgf)  # scale mask to balence gradients
        dflow = self.flow_head(imgf)
        coords1 = coords1 + dflow

        return mask, coords1



#helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    return lambda *args: val

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def group_images_by_max_seq_len(
    images: List[Tensor],
    patch_size: int,
    calc_token_dropout = None,
    max_seq_len = 2048

) -> List[List[Tensor]]:

    calc_token_dropout = default(calc_token_dropout, always(0.))

    groups = []
    group = []
    seq_len = 0

    if isinstance(calc_token_dropout, (float, int)):
        calc_token_dropout = always(calc_token_dropout)

    for image in images:
        assert isinstance(image, Tensor)

        image_dims = image.shape[-2:]
        ph, pw = map(lambda t: t // patch_size, image_dims)

        image_seq_len = (ph * pw)
        image_seq_len = int(image_seq_len * (1 - calc_token_dropout(*image_dims)))

        assert image_seq_len <= max_seq_len, f'image with dimensions {image_dims} exceeds maximum sequence length'

        if (seq_len + image_seq_len) > max_seq_len:
            groups.append(group)
            group = []
            seq_len = 0

        group.append(image)
        seq_len += image_seq_len

    if len(group) > 0:
        groups.append(group)

    return groups

#normalization
#they use layernorm without bias, something that pytorch does not offer
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

#they use a query-key normalization that is equivalent to rms norm (no mean-centering, learned gamma), from vit 22B paper
class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma

#feedforward
def FeedForward(dim, hidden_dim, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )

#attention
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
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




    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_mask = None
    ):
        x = self.norm(x)
        kv_input = default(context, x)

        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv
        )

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

#transformer block
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

    def forward(
        self,
        x,
        mask = None,
        attn_mask = None
    ):
        for attn, ff in self.layers:
            x = attn(x, mask = mask, attn_mask = attn_mask) + x
            x = ff(x) + x

        return self.norm(x)

class NaViT(nn.Module):
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
        self.decoder_pred_ft = nn.Linear(dim, patch_size ** 2 * 2, bias=True)  # decoder to patch

        self.update_block = UpdateBlock(patch_size ** 2 * 2)

        #what percent of tokens to dropout
        #if int or float given, then assume constant dropout prob
        #otherwise accept a callback that in turn calculates dropout prob from height and width

        self.calc_token_dropout = None

        if callable(token_dropout_prob):
            self.calc_token_dropout = token_dropout_prob

        elif isinstance(token_dropout_prob, (float, int)):
            # assert 0. < token_dropout_prob < 1.
            token_dropout_prob = float(token_dropout_prob)
            self.calc_token_dropout = lambda height, width: token_dropout_prob

        #calculate patching related stuff

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

        #final attention pooling queries

        self.attn_pool_queries = nn.Parameter(torch.randn(dim))
        self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)

        #output to logits

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_classes, bias = False)
        )

    @property
    def device(self):
        return next(self.parameters()).device


    
    #add
    def unpatchify(self, x, ph, pw):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = 16
        # h = w = int(x.shape[1] ** .5)
        # assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], ph, pw, p*p*2))

        return x

    def coords_grid(self,batch, ht, wd):
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
        mask = mask.view(N, 1, 9, 16, 16, H, W) #change to 16
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(16 * flow, [3, 3], padding=1)#change to 16
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 16 * H, 16 * W) #change to 16        

    def forward(
        self,
        batched_images: Union[List[Tensor], List[List[Tensor]]], #assume different resolution images already grouped correctly
        group_images = False,
        group_max_seq_len = 2048
    ):
        p, c, device, has_token_dropout = self.patch_size, self.channels, self.device, exists(self.calc_token_dropout)

        arange = partial(torch.arange, device = device)

        pad_sequence = partial(orig_pad_sequence, batch_first = True)


        #auto pack if specified

        if group_images:
            batched_images = group_images_by_max_seq_len(
                batched_images,
                patch_size = self.patch_size,
                calc_token_dropout = self.calc_token_dropout,
                max_seq_len = group_max_seq_len
            )

        #process images into variable lengthed sequences with attention mask

        num_images = []
        batched_sequences = []
        batched_positions = []
        batched_image_ids = []

        for images in batched_images:
            num_images.append(len(images))
            image_bak = images[0].unsqueeze(0)
            #print('image0',images[0].shape)

            sequences = []
            positions = []
            image_ids = torch.empty((0,), device = device, dtype = torch.long)

            for image_id, image in enumerate(images):
                assert image.ndim ==3 and image.shape[0] == c
                image_dims = image.shape[-2:]
                assert all([divisible_by(dim, p) for dim in image_dims]), f'height and width {image_dims} of images must be divisible by patch size {p}'

                ph, pw = map(lambda dim: dim // p, image_dims)

                pos = torch.stack(torch.meshgrid((
                    arange(ph),
                    arange(pw)
                ), indexing = 'ij'), dim = -1)

                pos = rearrange(pos, 'h w c -> (h w) c')
                seq = rearrange(image, 'c (h p1) (w p2) -> (h w) (c p1 p2)', p1 = p, p2 = p)

                seq_len = seq.shape[-2]

                if has_token_dropout:
                    token_dropout = self.calc_token_dropout(*image_dims)
                    num_keep = max(1, int(seq_len * (1 - token_dropout)))
                    keep_indices = torch.randn((seq_len,), device = device).topk(num_keep, dim = -1).indices

                    seq = seq[keep_indices]
                    pos = pos[keep_indices]

                image_ids = F.pad(image_ids, (0, seq.shape[-2]), value = image_id)
                sequences.append(seq)
                positions.append(pos)

            batched_image_ids.append(image_ids)
            batched_sequences.append(torch.cat(sequences, dim = 0))
            batched_positions.append(torch.cat(positions, dim = 0))

        #derive key padding mask

        lengths = torch.tensor([seq.shape[-2] for seq in batched_sequences], device = device, dtype = torch.long)
        max_length = arange(lengths.amax().item())
        key_pad_mask = rearrange(lengths, 'b -> b 1') <= rearrange(max_length, 'n -> 1 n')

        print('key_pad_mask',key_pad_mask.shape)

        #derive attention mask, and combine with key padding mask from above

        batched_image_ids = pad_sequence(batched_image_ids)
        attn_mask = rearrange(batched_image_ids, 'b i -> b 1 i 1') == rearrange(batched_image_ids, 'b j -> b 1 1 j')
        attn_mask = attn_mask & rearrange(key_pad_mask, 'b j -> b 1 1 j')
  
        print('attn_mask',attn_mask.shape)

        #combine patched images as well as the patched width / height positions for 2d positional embedding

        patches = pad_sequence(batched_sequences)
        patch_positions = pad_sequence(batched_positions)

        #need to know how many images for final attention pooling

        num_images = torch.tensor(num_images, device = device, dtype = torch.long)        

        #to patches
        print('patches',patches.shape)
        x = self.to_patch_embedding(patches)        

        #factorized 2d absolute positional embedding

        h_indices, w_indices = patch_positions.unbind(dim = -1)

        h_pos = self.pos_embed_height[h_indices]
        w_pos = self.pos_embed_width[w_indices]

        x = x + h_pos + w_pos

        #embed dropout

        x = self.dropout(x)

        #attention

        x = self.transformer(x, attn_mask = attn_mask)
        print('transformer,x',x.shape)
        # x = x.
        # return x

        x = self.decoder_norm_ft(x)
        # return x

        x = self.decoder_pred_ft(x)
        x = self.unpatchify(x,ph, pw)

        print('x', x.shape)  # torch.Size([1, 18, 18, 512])

        coodslar, coords0, coords1 = self.initialize_flow(image_bak)
        coords1 = coords1.detach()
        # print(coodslar, coords0.shape, coords1.shape)
        mask, coords1 = self.update_block(x, coords1)
        flow_up = self.upsample_flow(coords1 - coords0, mask)
        #print(coodslar.shape,flow_up.shape)
        bm_up = coodslar + flow_up

        return bm_up





        # #do attention pooling at the end

        # max_queries = num_images.amax().item()

        # queries = repeat(self.attn_pool_queries, 'd -> b n d', n = max_queries, b = x.shape[0])

        # #attention pool mask

        # image_id_arange = arange(max_queries)

        # attn_pool_mask = rearrange(image_id_arange, 'i -> i 1') == rearrange(batched_image_ids, 'b j -> b 1 j')

        # attn_pool_mask = attn_pool_mask & rearrange(key_pad_mask, 'b j -> b 1 j')

        # attn_pool_mask = rearrange(attn_pool_mask, 'b i j -> b 1 i j')

        # #attention pool

        # x = self.attn_pool(queries, context = x, attn_mask = attn_pool_mask) + queries

        # x = rearrange(x, 'b n d -> (b n) d')
        # print('x',x.shape)

        # return x

        # # #each batch element may not have same amount of images

        # # is_images = image_id_arange < rearrange(num_images, 'b -> b 1')
        # # is_images = rearrange(is_images, 'b n -> (b n)')
        # # print('is_images',is_images.shape)

        # # x = x[is_images]

        # # #project out to logits

        # # x = self.to_latent(x)

        # # return self.mlp_head(x)
        return x
