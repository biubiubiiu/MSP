import torch
import torch.nn as nn


def conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    padding = kernel_size // 2 if dilation == 1 else dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, dilation=dilation)


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), dilation=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias, dilation=dilation))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class EnHead1(nn.Module):
    def __init__(self, n_view=5, n_feats=32):
        super(EnHead1, self).__init__()

        self.down0 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(n_view * 3, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.down = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(n_view * 3, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.conv_low_cat = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResBlock(n_feats * 2, kernel_size=3),
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(n_feats * 2, n_feats, kernel_size=2, stride=2)
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResBlock(n_feats * 2, kernel_size=3),
            nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.out = nn.Sequential(
            nn.Conv2d(n_feats, n_view * 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        _, _, h, w = x.shape
        x_low = self.down0(x)
        # x: [N*an, an*c, h, w]
        feats = self.conv(x)
        down = self.down(feats)

        feats_low = self.conv_1(x_low)
        low_cat = torch.cat([feats_low, down], dim=1)
        low_cat = self.conv_low_cat(low_cat)
        up = self.up(low_cat)
        high_cat = torch.cat([feats, up], dim=1)

        feats = self.conv_last(high_cat)

        out = self.out(feats)
        return feats, out


class EnBody(nn.Module):
    def __init__(self, n_view=5, n_feats=32):
        super(EnBody, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_view * 3, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(n_feats, kernel_size=3),
            ResBlock(n_feats, kernel_size=3),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(n_feats, n_view * 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats_pre, fusion):
        # feats_pre: [N*an, 32, h, w],
        # fusion: [N*an, an*c, h, w]
        feats_fusion = self.encoder(fusion)
        feats = torch.cat([feats_pre, feats_fusion], dim=1)
        feats = self.conv(feats)
        out = self.decoder(feats)
        return feats, out


class EnTail(nn.Module):
    def __init__(self, n_view=5, n_feats=32):
        super(EnTail, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(n_view * 3, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(n_feats, kernel_size=3),
            ResBlock(n_feats, kernel_size=3),
            nn.Conv2d(n_feats, n_view * 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats_pre, fusion):
        feats_fusion = self.encoder(fusion)
        feats = torch.cat([feats_pre, feats_fusion], dim=1)
        return self.conv(feats)


class EnCenterHead1(nn.Module):
    def __init__(self, n_feats=32, n_view=5):
        super(EnCenterHead1, self).__init__()

        self.an = n_view
        self.down0 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(9 * 3, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(9 * 3, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(9 * 1, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.conv_low_cat = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResBlock(n_feats * 2, kernel_size=3),
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(n_feats * 2, n_feats, kernel_size=2, stride=2)
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResBlock(n_feats * 2, kernel_size=3),
            nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.out = nn.Sequential(
            nn.Conv2d(n_feats, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def prepare_data(self, lf):
        N, an2, c, h, w = lf.shape
        an = self.an
        x = lf.reshape(N, an * an, c, h * w)

        x = x.reshape(N, an * an, c, h * w)
        x = torch.transpose(x, 1, 3)
        x = x.reshape(N, h * w, c, an, an)
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), value=0)

        x = x.reshape(N, h * w, c, (an + 2) * (an + 2))
        x = torch.transpose(x, 1, 3)
        x = x.reshape(N, (an + 2) * (an + 2), c, h, w)
        x = x.reshape(N, (an + 2), (an + 2), c, h, w)

        x6 = x[:, :-2, :-2].reshape(N, an2, c, h, w)
        x2 = x[:, :-2, 1:-1].reshape(N, an2, c, h, w)
        x8 = x[:, :-2, 2:].reshape(N, an2, c, h, w)
        x4 = x[:, 1:-1, :-2].reshape(N, an2, c, h, w)

        x3 = x[:, 1:-1, 2:].reshape(N, an2, c, h, w)
        x7 = x[:, 2:, :-2].reshape(N, an2, c, h, w)
        x1 = x[:, 2:, 1:-1].reshape(N, an2, c, h, w)
        x5 = x[:, 2:, 2:].reshape(N, an2, c, h, w)
        focal_stack = torch.cat([x6, x2, x8, x4, lf, x3, x7, x1, x5], dim=2)
        return focal_stack

    def forward(self, lf):
        N, an2, c, h, w = lf.shape
        fs = self.prepare_data(lf)
        x = fs.reshape(N * an2, c * 9, h, w)
        _, _, h, w = x.shape
        x_low = self.down0(x)
        # x: [N*an, an*c, h, w]
        feats = self.conv(x)
        down = self.down(feats)

        feats_low = self.conv_1(x_low)
        low_cat = torch.cat([feats_low, down], dim=1)
        low_cat = self.conv_low_cat(low_cat)
        up = self.up(low_cat)
        high_cat = torch.cat([feats, up], dim=1)

        feats = self.conv_last(high_cat)

        out = self.out(feats)
        return feats, out


class EnCenterBody(nn.Module):
    def __init__(self, n_feats=32, n_view=5, is_tail=False):
        super(EnCenterBody, self).__init__()
        self.an = n_view

        self.encoder = nn.Sequential(
            nn.Conv2d(27, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResBlock(n_feats, kernel_size=3),
            ResBlock(n_feats, kernel_size=3),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(n_feats, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def prepare_data1(self, lf):
        N, an2, c, h, w = lf.shape
        an = self.an
        focal_stack = torch.zeros((N, an, an, c * 9, h, w)).to(lf.device)
        x = lf.reshape(N, an * an, c, h * w)

        x = x.reshape(N, an * an, c, h * w)
        x = torch.transpose(x, 1, 3)
        x = x.reshape(N, h * w, c, an, an)
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), value=0)

        x = x.reshape(N, h * w, c, (an + 2) * (an + 2))
        x = torch.transpose(x, 1, 3)
        x = x.reshape(N, (an + 2) * (an + 2), c, h, w)
        x = x.reshape(N, (an + 2), (an + 2), c, h, w)

        x6 = x[:, :-2, :-2].reshape(N, an2, c, h, w)
        x2 = x[:, :-2, 1:-1].reshape(N, an2, c, h, w)
        x8 = x[:, :-2, 2:].reshape(N, an2, c, h, w)
        x4 = x[:, 1:-1, :-2].reshape(N, an2, c, h, w)

        x3 = x[:, 1:-1, 2:].reshape(N, an2, c, h, w)
        x7 = x[:, 2:, :-2].reshape(N, an2, c, h, w)
        x1 = x[:, 2:, 1:-1].reshape(N, an2, c, h, w)
        x5 = x[:, 2:, 2:].reshape(N, an2, c, h, w)
        focal_stack = torch.cat([x6, x2, x8, x4, lf, x3, x7, x1, x5], dim=2)

        return focal_stack

    def forward(self, lf, feats):
        N, an2, c, h, w = lf.shape

        fs = self.prepare_data1(lf)

        fs = fs.reshape(N * an2, c * 9, h, w)
        x = self.encoder(fs)

        x = torch.cat([x, feats], dim=1)
        x = self.conv(x)
        out = self.decoder(x)

        return x, out


class LFEn_s3(nn.Module):
    def __init__(self, n_view=5):
        super(LFEn_s3, self).__init__()
        self.an = n_view
        n_feats = 32

        self.head_0, self.head_90 = EnHead1(n_view, n_feats), EnHead1(n_view, n_feats)
        self.body_0, self.body_90 = EnBody(n_view, n_feats), EnBody(n_view, n_feats)
        self.tail_0, self.tail_90 = EnTail(n_view, n_feats), EnTail(n_view, n_feats)

        self.c_head = EnCenterHead1(n_feats, n_view)
        self.c_body_0 = EnCenterBody(n_feats, n_view)
        self.c_body_3 = EnCenterBody(n_feats, n_view, is_tail=True)

    def my_norm(self, x):
        N, an2, c, h, w = x.shape
        lf_avg = torch.mean(x, dim=1, keepdim=False)  # [N, c, h, w]
        gray = 0.2989 * lf_avg[:, 0, :, :] + 0.5870 * lf_avg[:, 1, :, :] + 0.1140 * lf_avg[:, 2, :, :]  # [N, h, w]
        temp = (1 - gray) * gray
        ratio = (h * w) / (2 * torch.sum(temp.reshape(N, -1), dim=1))
        return ratio

    def prepare_data(self, x):
        N, an2, c, h, w = x.shape

        x = x.view(N, self.an, self.an, c, h, w)
        x_0 = x.view(N * self.an, self.an, c, h, w)
        x_0 = x_0.reshape(N * self.an, self.an * c, h, w)

        x_90 = torch.transpose(x, 1, 2)
        x_90 = x_90.reshape(N * self.an, self.an, c, h, w)
        x_90 = x_90.reshape(N * self.an, self.an * c, h, w)
        return x_0, x_90

    def post_process(self, out_0, out_90, x):
        N, an2, c, h, w = x.shape
        # [N*an, 3*an, h, w]
        out_0 = out_0.view(N * self.an, self.an, c, h, w)
        out_0 = out_0.view(N, self.an, self.an, c, h, w)
        out_0 = out_0.view(N, an2, c, h, w)

        out_90 = out_90.view(N * self.an, self.an, c, h, w)
        out_90 = out_90.view(N, self.an, self.an, c, h, w)
        out_90 = torch.transpose(out_90, 1, 2).reshape(N, an2, c, h, w)
        return out_0, out_90

    def forward(self, x):
        N, an2, c, h, w = x.shape
        ratio = self.my_norm(x).reshape(N, 1, 1, 1, 1).expand_as(x)
        x = x * ratio

        # stage1
        c_feats_0, central_view_0 = self.c_head(x)
        x_0, x_90 = self.prepare_data(x)  # [N*an, an*c, h, w]
        feats_0, head_0 = self.head_0(x_0)
        feats_90, head_90 = self.head_90(x_90)
        head_0, head_90 = self.post_process(head_0, head_90, x)  # [N, an2, c, h, w]
        central_view_0 = central_view_0.reshape(N, an2, c, h, w)
        out1 = (head_0 + head_90 + central_view_0) / 3

        # stage2
        c_feats_1, central_view_1 = self.c_body_0(out1, c_feats_0)
        x_0, x_90 = self.prepare_data(out1)
        feats_0, body_0 = self.body_0(feats_0, x_0)
        feats_90, body_90 = self.body_90(feats_90, x_90)
        body_0, body_90 = self.post_process(body_0, body_90, x)
        central_view_1 = central_view_1.reshape(N, an2, c, h, w)
        out2 = (body_0 + body_90 + central_view_1) / 3

        # stage3
        _, central_view_4 = self.c_body_0(out2, c_feats_1)
        x_0, x_90 = self.prepare_data(out2)
        tail_0 = self.tail_0(feats_0, x_0)
        tail_90 = self.tail_90(feats_90, x_90)
        tail_0, tail_90 = self.post_process(tail_0, tail_90, x)
        central_view_4 = central_view_4.reshape(N, an2, c, h, w)
        out = (tail_0 + tail_90 + central_view_4) / 3
        return out1, out2,  out


if __name__ == '__main__':
    net = LFEn_s3()

    ckpt = torch.load('weights/1_20/weights.pkl', map_location='cpu')
    net.load_state_dict(ckpt)

    inp = torch.randn(1, 25, 3, 64, 64)
    print(net(inp)[-1].shape)
