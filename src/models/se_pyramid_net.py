import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.shake_drop import ShakeDrop
import utils
from models.se import SELayer

class ShakeBasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, p_shakedrop=1.0, use_shake_drop=True, use_se=True):
        super(ShakeBasicBlock, self).__init__()
        self.downsampled = stride == 2
        self.branch = self._make_branch(in_ch, out_ch, stride=stride, use_se=use_se)
        self.shortcut = not self.downsampled and None or nn.AvgPool2d(2, padding=1)
        if use_shake_drop:
            self.shake_drop = ShakeDrop(p_shakedrop)
        self.use_shake_drop = use_shake_drop

    def forward(self, x):
        h = self.branch(x)
        if self.use_shake_drop:
            h = self.shake_drop(h)
        h0 = x if not self.downsampled else self.shortcut(x)
        pad_zero = torch.zeros((h0.size(0), h.size(1) - h0.size(1), h0.size(2), h0.size(3)), dtype=x.dtype,
                device=x.device)
        h0 = torch.cat([h0, pad_zero], dim=1)

        return h + h0

    def _make_branch(self, in_ch, out_ch, stride=1, use_se=True, out_channel_ratio=4):
        seq = [nn.BatchNorm2d(in_ch),
               nn.Conv2d(in_ch, out_ch, 1, padding=0, stride=1, bias=False),
               nn.BatchNorm2d(out_ch),
               nn.ReLU(inplace=True),
               nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=stride, bias=False),
               nn.BatchNorm2d(out_ch),
               nn.ReLU(inplace=True),
               nn.Conv2d(out_ch, out_ch * out_channel_ratio, 1, padding=0, stride=1, bias=False),
               nn.BatchNorm2d(out_ch * out_channel_ratio)]
        if use_se:
            seq.append(SELayer(out_ch))
        return nn.Sequential(*seq)

class ShakePyramidNet(nn.Module):
    def __init__(self, depth, alpha, classes, input_channels, use_shake_drop, use_se, dropout,
            criterion=nn.CrossEntropyLoss()):
        super().__init__()
        self.use_shake_drop = use_shake_drop
        self.use_se = use_se
        self.criterion = criterion
        init_ch = 16
        # for BasicBlock
        n_units = (depth - 2) // 9
        init_chs = [init_ch] + [init_ch + math.ceil((alpha / (3 * n_units)) * (i + 1)) \
                for i in range(3 * n_units)]
        block = ShakeBasicBlock

        self.init_chs, self.u_idx = init_chs, 0
        self.ps_shakedrop = [1 - (1.0 - (0.5 / (3 * n_units)) * (i + 1)) for i in range(3 * n_units)]

        head_channels = 64

        #standard large image head
        self.head = nn.Sequential(
                nn.Conv2d(input_channels, head_channels, 3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(head_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_channels, head_channels, 3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(head_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_channels, init_ch, 3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(init_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3,  stride=2))

        self.layer1 = self._make_layer(n_units, block, 1)
        self.layer2 = self._make_layer(n_units, block, 2)
        self.layer3 = self._make_layer(n_units, block, 2)
        self.bn_out = nn.BatchNorm2d(init_chs[-1])
        self.fc_out = nn.Linear(init_chs[-1], classes)
        self.dropout = nn.Dropout(dropout)

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
        h = self.head(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = F.relu(self.bn_out(h))
        h = F.adaptive_avg_pool2d(h, (1, 1))
        h = h.view(h.size(0), -1)
        h = self.dropout(h)
        h = self.fc_out(h)
        return h

    def _make_layer(self, n_units, block, stride=1):
        layers = []
        for i in range(int(n_units)):
            layers.append(block(self.init_chs[self.u_idx], self.init_chs[self.u_idx+1],
                                stride, self.ps_shakedrop[self.u_idx], self.use_shake_drop, self.use_se))
            self.u_idx, stride = self.u_idx + 1, 1

        return nn.Sequential(*layers)

    def loss(self, logits, target):
        return self.criterion(logits, target)

    def top_k(self, pred, true, top_k):
        return utils.accuracy(pred, true, top_k)

