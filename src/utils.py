import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

def numpy2tensor(img):
    x0 = torch.from_numpy(img.copy()).float().cuda() / 255.0 * 2.0 - 1.
    x0 = torch.stack([x0], dim=0)
    # einops.rearrange(x0, 'b h w c -> b c h w').clone()
    return x0.permute(0, 3, 1, 2)

def pil2tensor(img):
    return numpy2tensor(np.array(img))

def tensor2numpy(img):
    image = (img / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    return images

def tensor2pil(img):
    return Image.fromarray(tensor2numpy(img)[0])

def cv2sod(img):
    in_ = np.array(img, dtype=np.float32)
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2,0,1))
    image = torch.Tensor(in_)
    return F.interpolate(image.unsqueeze(0), scale_factor=0.5, mode='bilinear')

def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def visualize(img_arr, dpi):
    plt.figure(figsize=(10,10),dpi=dpi)
    plt.imshow(((img_arr.detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')
    plt.show()


def calc_mean_std(feat, eps=1e-5, chunk=1):
    size = feat.size()
    assert (len(size) == 4)
    if chunk == 2:
        feat = torch.cat(feat.chunk(2), dim=3)
    N, C = size[:2]
    feat_var = feat.view(N//chunk, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N//chunk, C, -1).mean(dim=2).view(N//chunk, C, 1, 1)
    return feat_mean.repeat(chunk,1,1,1), feat_std.repeat(chunk,1,1,1)


def adaptive_instance_normalization(content_feat, style_feat, chunk=1):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat, chunk)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class Dilate():
    def __init__(self, kernel_size=7, channels=1, device='cpu'):
        self.kernel_size=kernel_size
        self.channels = channels
        gaussian_kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)
        self.mean = (self.kernel_size - 1)//2
        gaussian_kernel = gaussian_kernel.to(device)
        self.gaussian_filter = gaussian_kernel
        
    def __call__(self, x):
        x = F.pad(x, (self.mean,self.mean,self.mean,self.mean), "replicate")
        return torch.clamp(F.conv2d(x, self.gaussian_filter, bias=None), 0, 1)

@torch.no_grad()
def get_saliency(imgs, sod_model, dilate):
    imgs_sod = torch.cat([cv2sod(img) for img in imgs], dim=0).cuda()
    _, _, up_sal_f = sod_model(imgs_sod)
    saliency = 1-dilate(np.squeeze(torch.sigmoid(up_sal_f[-1])).unsqueeze(1))
    del up_sal_f
    torch.cuda.empty_cache()
    return saliency
