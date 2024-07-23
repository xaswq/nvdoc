import random
import torch
import torch.utils.data as data
import cv2
import numpy as np
from PIL import Image

def color_jitter(im, brightness=0., contrast=0., saturation=0., hue=0.):
    f = random.uniform(1 - contrast, 1 + contrast)
    im = np.clip(im * f, 0., 1.)
    f = random.uniform(-brightness, brightness)
    im = np.clip(im + f, 0., 1.).astype(np.float32)
    
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    f = random.uniform(-hue, hue)
    hsv[0] = np.clip(hsv[0] + f * 360, 0., 360.)
    f = random.uniform(-saturation, saturation)
    hsv[2] = np.clip(hsv[2] + f, 0., 1.)
    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    im = np.clip(im, 0., 1.)
    return im

def tight_crop(im, wc):
    msk = ((wc[:, :, 0] != 0) & (wc[:, :, 1] != 0) & (wc[:, :, 2] != 0)).astype(np.uint8)
    [y, x] = msk.nonzero()
    minx, maxx, miny, maxy = min(x), max(x), min(y), max(y)
    
    im = im[miny: maxy + 1, minx: maxx + 1, :]
    
    s = 20
    im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
    
    cx1 = random.randint(5, 40)
    cx2 = random.randint(5, 40) + 1
    cy1 = random.randint(5, 40)
    cy2 = random.randint(5, 40) + 1
    
    im = im[cy1: -cy2, cx1: -cx2, :]
    
    t = miny - s + cy1
    b = 448 - maxy - s + cy2
    l = minx - s + cx1
    r = 448 - maxx - s + cx2
    
    return im, t, b, l, r

def data_aug(im, wc):
    im = im / 255.0
    im, t, b, l, r = tight_crop(im, wc)
    im = color_jitter(im, 0.2, 0.2, 0.2, 0.2)
    return im, t, b, l, r

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

class Dataset(data.Dataset):
    def __init__(self, mode):
        self.mode = mode
        
        self.img_list = sorted(eval(open('/aiarena/group/mmitgroup/liusk/upload/img_tra_val.txt', 'r').read())['train']) + \
                        sorted(eval(open('/aiarena/group/mmitgroup/liusk/upload/img_tra_val.txt', 'r').read())['val'])
        self.bm_list = sorted(eval(open('/aiarena/group/mmitgroup/liusk/upload/bm_tra_val.txt', 'r').read())['train']) + \
                       sorted(eval(open('/aiarena/group/mmitgroup/liusk/upload/bm_tra_val.txt', 'r').read())['val'])
        self.wc_list = sorted(eval(open('/aiarena/group/mmitgroup/liusk/upload/wc_tra_val.txt', 'r').read())['train']) + \
                       sorted(eval(open('/aiarena/group/mmitgroup/liusk/upload/wc_tra_val.txt', 'r').read())['val'])
        
        assert len(self.img_list) == len(self.bm_list) == len(self.wc_list)

    def __getitem__(self, index):
        im = np.array(Image.open(self.img_list[index]))[:, :, :3]  # 0-255
        wc = cv2.imread(self.wc_list[index], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        
        # deal inputs
        im, t, b, l, r = data_aug(im, wc)
        
        # 使用新的resize函数
        im = resize_keep_aspect_ratio(im)
        h, w = im.shape[:2]
        
        im = torch.from_numpy(im).permute(2, 0, 1).float()
        
        # deal gt
        bm = np.load(self.bm_list[index]).transpose(0, 2, 1)  # ground truth
        bm[0] = (bm[0] - l - 1.01) / (448.0 - l - r) * (w - 2)
        bm[1] = (bm[1] - t - 1.01) / (448.0 - t - b) * (h - 2)
        
        bmm = (bm[0] >= 0) & (bm[1] >= 0) & (bm[0] <= w-1) & (bm[1] <= h-1)
        [y, x] = bmm.nonzero()
        minx, maxx, miny, maxy = min(x), max(x), min(y), max(y)
        
        bm00 = bm[0, miny: maxy + 1, minx: maxx + 1]
        bm11 = bm[1, miny: maxy + 1, minx: maxx + 1]
        
        bm0 = cv2.resize(bm00, (w, h))
        bm1 = cv2.resize(bm11, (w, h))
        
        lbl = np.stack([bm0, bm1], axis=0)
        lbl = torch.from_numpy(lbl).float()
        
        return im, lbl

    def __len__(self):
        return len(self.img_list)