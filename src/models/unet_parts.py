# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.scse import SCSE
from models.shake_drop import ShakeDrop

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, use_scse, scse_ratio):
        super().__init__()
        seq = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)]

        if use_scse:
            seq.append(SCSE(out_ch, scse_ratio))

        self.branch = nn.Sequential(*seq)

    def forward(self, x):
        return self.branch(x)

class pyramid_bottleneck(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, downsampled, use_scse=True, scse_ratio=8,
            use_shake_drop=False, p_shakedrop=1.):
        super().__init__()
        stride = 2 if downsampled else 1
        seq = [nn.BatchNorm2d(in_ch),
               nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False),
               nn.BatchNorm2d(out_ch),
               nn.ReLU(inplace=True),
               nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1, bias=False),
               nn.BatchNorm2d(out_ch)]
        if use_scse:
            seq.append(SCSE(out_ch, scse_ratio))

        self.branch = nn.Sequential(*seq)

        self.downsampled = downsampled
        self.shortcut = not self.downsampled and None or nn.AvgPool2d(2, padding=0)
        if use_shake_drop:
            self.shake_drop = ShakeDrop(p_shakedrop)
        self.use_shake_drop = use_shake_drop

    def forward(self, x):
        h = self.branch(x)
        if self.use_shake_drop:
            h = self.shake_drop(h)
        h0 = x if not self.downsampled else self.shortcut(x)
        pad_zero = torch.zeros((h0.size(0), abs(h.size(1) - h0.size(1)), h0.size(2), h0.size(3)), dtype=x.dtype,
                device=x.device)
        if h.size(1) > h0.size(1):
            h0 = torch.cat([h0, pad_zero], dim=1)
        else:
            h = torch.cat([h, pad_zero], dim=1)


        return h + h0

class down(nn.Module):
    def __init__(self, in_ch, out_chs, p_shakedrops, use_scse=True, scse_ratio=8, 
            use_shake_drop=False):
        super(down, self).__init__()
        bottlenecks = []
        assert len(out_chs) == len(p_shakedrops)

        for out_ch, p_shakedrop in zip(out_chs, p_shakedrops):
            bottlenecks.append(pyramid_bottleneck(in_ch, out_ch, out_ch == out_chs[-1],
            use_scse, scse_ratio, use_shake_drop, p_shakedrop))
            in_ch = out_ch

        self.net = nn.Sequential(*bottlenecks)

    def forward(self, x):
        x = self.net(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False, use_scse=True, scse_ratio=8):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

        self.net = double_conv(in_ch, out_ch, use_scse, scse_ratio)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        x = torch.cat([x2, x1], dim=1)
        x = self.net(x)
        return x
