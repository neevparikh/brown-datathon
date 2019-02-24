import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from models.unet_parts import up, down
from models.custom_loss import lovasz_hinge

class UNet(nn.Module):
    def __init__(self, alpha, classes, input_channels, use_shake_drop, use_scse, 
            num_downsamples, num_blocks_per_downsample):
        super().__init__()
        self.use_shake_drop = use_shake_drop
        self.use_scse = use_scse
        in_ch = init_ch = 16
        # for BasicBlock
        n_units = num_downsamples * num_blocks_per_downsample
        add_rate = alpha / n_units
        p_add_rate = 0.5 / n_units
        head_channel_ratio = 4
        scse_ratio = 8

        p = 1.

        self.head = nn.Sequential(
                nn.Conv2d(input_channels, in_ch * head_channel_ratio, 3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(in_ch * head_channel_ratio),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch * head_channel_ratio, in_ch, 3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True))

        self.down_blocks = nn.ModuleList()

        down_chs = []

        for i in range(num_downsamples):
            out_chs = []
            p_shakedrops = []
            next_ch = in_ch 
            down_chs.append(round(in_ch))
            for j in range(num_blocks_per_downsample):
                next_ch += add_rate 
                out_chs.append(round(next_ch))
                p_shakedrops.append(p)
                p -= p_add_rate

            self.down_blocks.append(down(round(in_ch), out_chs, p_shakedrops, use_scse, 
                scse_ratio, use_shake_drop))

            in_ch = next_ch

        assert init_ch + alpha == in_ch 
        assert abs(p - 0.5) < 1e-5

        self.up_blocks = nn.ModuleList()

        n_units = num_downsamples
        add_rate = alpha / n_units

        for i in range(num_downsamples):
            out_chs = []
            p_shakedrops = []
            next_ch = in_ch 
            next_ch -= add_rate 

            self.up_blocks.append(up(round(in_ch) + down_chs[-i -1], round(next_ch), 
                True, use_scse, scse_ratio))
            in_ch = next_ch

        assert init_ch == in_ch 

        self.end = nn.Conv2d(round(in_ch), classes, 1)
        

        # Initialize paramters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.head(x)
        down_x = []

        for block in self.down_blocks:
            down_x.append(x)
            x = block(x)

        assert len(self.up_blocks) == len(down_x)

        down_x = list(reversed(down_x))

        for block, d_x in zip(self.up_blocks, down_x):
            x = block(x, d_x)

        return torch.sigmoid(self.end(x))

    def loss(self, pred, target):
        return lovasz_hinge(pred, target)

    def iou(self, pred, target):
        return utils.iou(pred.contiguous(), target.contiguous(), n_classes=1)
