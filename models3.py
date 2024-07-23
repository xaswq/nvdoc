from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

import torch.nn.functional as F


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



class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches


        # why?
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #+1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim),
                                      requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,   norm_layer=norm_layer)
            for i in range(depth)])
        # self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        # self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        #
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        #
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
        #                                       requires_grad=False)  # fixed sin-cos embedding

        # self.decoder_blocks = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])

        self.decoder_norm_ft = norm_layer(embed_dim)
        self.decoder_pred_ft = nn.Linear(embed_dim, patch_size ** 2 * 2, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        # bm tail
        self.update_block = UpdateBlock(patch_size ** 2 * 2)
        
        # self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()
    
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True) #change to True
        # print(pos_embed.shape)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        #
        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
        #                                             int(self.patch_embed.num_patches ** .5), cls_token=False)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
    
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)
    
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p*p*2))
        # x = torch.einsum('nhwpqc->nchpwq', x)
        # imgs = x.reshape(shape=(x.shape[0], 2, h * p, h * p))
        return x



    def initialize_flow(self, img):
        N, C, H, W = img.shape
        coodslar = coords_grid(N, H, W).to(img.device)
        coords0 = coords_grid(N, H // 16, W // 16).to(img.device)
        coords1 = coords_grid(N, H // 16, W // 16).to(img.device)

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

    def forward(self, imgs):

        x = self.patch_embed(imgs)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # x = self.norm(x)
        x = x[:,1:,:]

        x = self.decoder_norm_ft(x)
        x = self.decoder_pred_ft(x)
        x = self.unpatchify(x)
        #print('x', x.shape)  # torch.Size([1, 9, 9, 2048])

        coodslar, coords0, coords1 = self.initialize_flow(imgs)
        coords1 = coords1.detach()
        # print(coodslar, coords0.shape, coords1.shape)
        mask, coords1 = self.update_block(x, coords1)
        flow_up = self.upsample_flow(coords1 - coords0, mask)
        #print(coodslar.shape,flow_up.shape)
        bm_up = coodslar + flow_up

        return bm_up


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


#############################################

#  change to 16
def mae_vit_mini_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=288,
                                 patch_size=16, embed_dim=512, depth=10, num_heads=8,
                                 # decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
                                 mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    net = mae_vit_mini_patch16_dec512d8b().cuda()
    print(get_parameter_number(net))  # 69090500 加attention后69442032
    with torch.no_grad():
        inputs = torch.zeros(1, 3, 288, 288).cuda()
        outs = net(inputs)
        print(outs.shape)  # torch.Size([2, 3, 256, 256]) torch.Size([2, 2, 256, 256])
