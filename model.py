import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from kornia.color import rgb_to_y


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(channels, channels, kernel_size=3, padding=1))

    def forward(self, x):
        out = x + self.body(x)
        return out


class VSFEM(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(VSFEM, self).__init__()

        self.proj = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True))
        self.down = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(kernel_size=2, stride=2))
        self.up = nn.ConvTranspose2d(mid_channels * 2, mid_channels, kernel_size=2, stride=2)

        self.conv0 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(nn.Conv2d(mid_channels * 2, mid_channels * 2, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   ResBlock(mid_channels * 2))

        self.conv2 = nn.Sequential(nn.Conv2d(mid_channels * 2, mid_channels * 2, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   ResBlock(mid_channels * 2),
                                   nn.Conv2d(mid_channels * 2, mid_channels, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True))

        self.out = nn.Sequential(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        init_feat = self.proj(x)
        feat_ss = self.conv0(F.avg_pool2d(x, kernel_size=2, stride=2))
        feat_ss = self.conv1(torch.cat([feat_ss, self.down(init_feat)], dim=1))
        feat_ls = torch.cat([init_feat, self.up(feat_ss)], dim=1)
        feat = self.conv2(feat_ls)
        out = self.out(feat)
        return feat, out


class SIM(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(SIM, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True))

        self.body = nn.Sequential(nn.Conv2d(mid_channels * 2, mid_channels, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True),
                                  ResBlock(mid_channels),
                                  ResBlock(mid_channels))

        self.out = nn.Sequential(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                                 nn.ReLU(inplace=True))

    def forward(self, x, prev_feats):
        x = self.proj(x)
        x = torch.cat([x, prev_feats], dim=1)
        x = self.body(x)
        out = self.out(x)
        return x, out


class MSPNet(nn.Module):
    def __init__(self, cfg):
        super(MSPNet, self).__init__()

        resolution = cfg.resolution
        n_stage = cfg.stages or 3
        mid_channels = cfg.channels or 32

        self.n_stage = n_stage

        self.feat_hstack = nn.ModuleList([VSFEM(resolution * 3, mid_channels, resolution * 3)])
        self.feat_cstack = nn.ModuleList([VSFEM(9 * 3, mid_channels, 3)])
        self.feat_vstack = nn.ModuleList([VSFEM(resolution * 3, mid_channels, resolution * 3)])

        for _ in range(n_stage - 1):
            self.feat_hstack.append(SIM(resolution * 3, mid_channels, resolution * 3))
            self.feat_cstack.append(SIM(9 * 3, mid_channels, 3))
            self.feat_vstack.append(SIM(resolution * 3, mid_channels, resolution * 3))

    def estimate_ratio(self, x):
        intensity = torch.mean(x, dim=[1, 2], keepdim=False)
        brightness = rgb_to_y(intensity)
        t = ((1 - brightness) * brightness).mean(dim=[-1, -2], keepdim=False)
        ratio = 1 / (2 * t)  # [B, 1]
        return ratio

    def forward_single_stage(self, x, stage_idx, prev_outputs=None):
        # 3*3 stream input
        u, v = x.shape[1:3]
        pad_x = F.pad(x, (*(0, 0) * (x.ndim-3), *(1, 1) * 2, *(0, 0)),
                      mode='constant', value=0)  # [B, U+2, V+2, C, H, W]

        offsets = [0, 1, 2]
        slices = [pad_x[:, i:i+u, j:j+v] for i, j in itertools.product(offsets, offsets)]
        cs = torch.cat(slices, dim=3)
        cs = rearrange(cs, 'b u v (n c) h w -> (b u v) (n c) h w', n=len(offsets)**2)

        # horizontal stream input
        hs = rearrange(x, 'b u v c h w -> (b u) (v c) h w')

        # vertical stream input
        vs = rearrange(x, 'b u v c h w -> (b v) (u c) h w')

        if stage_idx == 0:
            feat_h, out_h = self.feat_hstack[stage_idx](hs)
            feat_c, out_c = self.feat_cstack[stage_idx](cs)
            feat_v, out_v = self.feat_vstack[stage_idx](vs)
        else:
            prev_feat_h, prev_feat_c, prev_feat_v = prev_outputs
            feat_h, out_h = self.feat_hstack[stage_idx](hs, prev_feat_h)
            feat_c, out_c = self.feat_cstack[stage_idx](cs, prev_feat_c)
            feat_v, out_v = self.feat_vstack[stage_idx](vs, prev_feat_v)

        out_h = rearrange(out_h, '(b u) (v c) h w -> b u v c h w', u=u, v=v)
        out_v = rearrange(out_v, '(b v) (u c) h w -> b u v c h w', u=u, v=v)
        out_c = rearrange(out_c, '(b u v) c h w -> b u v c h w', u=u, v=v)
        out = torch.stack([out_h, out_v, out_c], dim=0).mean(dim=0)
        return out, [feat_h, feat_c, feat_v]

    def forward(self, x):
        # pre-amplification
        ratio = self.estimate_ratio(x)
        x = x * ratio

        # multi-stage processing
        out, outs, hs = x, [], None
        for i in range(self.n_stage):
            out, hs = self.forward_single_stage(out, i, hs)
            outs.append(out)

        return outs


if __name__ == '__main__':
    from addict import Dict
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    cfg = Dict({'resolution': 5, 'stages': 3})
    net = MSPNet(cfg)

    inp = torch.randn(1, 5, 5, 3, 64, 64)
    flops = FlopCountAnalysis(net, inp)
    with open(f'summary.txt', 'w', encoding='utf-8') as f:
        f.write(flop_count_table(flops, max_depth=6))
    print(net(inp)[-1].shape)
