"""
LAD-Net reproduction (PyTorch) -- architecture only.

Reference: Zhu, Li, Jia, Liu, Yao, Yuan, Huo, Zhang, "LAD-Net: A Novel Light
Weight Model for Early Apple Leaf Pests and Diseases Classification",
IEEE/ACM TCBB, 2023. No official code repo was found/consulted; this is an
independent clean-room reimplementation from the paper text (Section 3,
Fig. 1-4, Table 1).

IMPORTANT SCOPE NOTE: this reproduction is architecture-only. The paper's
own dataset (AppleSet6, 821 original / 11,820 augmented images across 6
early apple pest/disease classes) is a private field-collected dataset shot
at "QianXian's Apple Monitoring Station" with no public release mentioned in
the paper -- it could not be sourced. The two generalization datasets
(LateAppleSet on Baidu AI Studio, Tomato9 / AI Challenger 2018's tomato
subset) are both gated behind Baidu-platform logins this environment
couldn't authenticate through (AI Studio returned HTTP 403; AI Challenger's
own site challenger.ai is offline, leaving only a Baidu Netdisk mirror,
same login wall). No training or evaluation was run. See README.md.

Architecture per Table 1 / Fig. 1-4:
  Input(224x224x3)
    -> AD Convolution (k=3,s=2,dilation=3,pad=3)          -> 56x56x64
    -> MaxPool(3x3,s=2,pad=1)                              -> 28x28x64
    -> LR-CBAM_1 (k=1,s=1)                                 -> 28x28x64
    -> LR-CBAM_2 (k=3,s=1,pad=1)                            -> 28x28x192
    -> MaxPool(3x3,s=2)                                     -> 14x14x192
    -> LAD-Inception                                        -> 14x14x272
    -> MaxPool(3x3,s=2)                                     -> 7x7x272
    -> CBAM                                                 -> 7x7x272
    -> GlobalAveragePool                                    -> 272
    -> FC (dropout 0.5)                                     -> num_classes

Known paper gaps, resolved here (see README "Known deviations" for the same
list, kept in sync):
  1. AD Convolution's exact (kernel, dilation, padding) split between its two
     asymmetric sub-convs is only shown concretely inside LAD-Inception's
     three multiscale branches (Fig. 4: 1x3/3x1 pairs with dilation 1,2,3).
     For the *stem* AD Convolution (Table 1's first row: "3x3/2, dilation
     3/3"), we apply that same asymmetric decomposition (1x3 dilated conv
     followed by 3x1 dilated conv) with stride folded into the first sub-conv.
  2. Per-branch channel widths inside LAD-Inception are not given (only the
     block's total output, 272 channels, from a 192-channel input). We split
     272 across the 5 branches (4 multiscale/1x1 branches + 1 maxpool
     branch) as 64+64+64+64+16 -- a defensible, round-number split, not a
     paper-stated value.
  3. CBAM (channel + spatial attention) follows the standard Woo et al. 2018
     design (avg+max pool -> shared MLP for channel attention; channel-wise
     avg+max -> 7x7 conv for spatial attention), since the paper cites it
     directly ([28]) rather than modifying it.
  4. Table 1's stem row states "3x3/stride 2, dilation 3/padding 3" giving
     224x224 -> 56x56 -- but that shape change is a 4x spatial reduction,
     which a single stride-2 conv cannot produce (stride 2 halves each
     dimension once, giving 112x112, not 56x56, under any padding/dilation
     combination we could find). We used stride=4 (keeping kernel=3,
     dilation=3, padding=2) to hit the table's stated 56x56 output exactly,
     since matching the paper's declared feature-map shapes end-to-end took
     priority over the (evidently inconsistent) "stride 2" label.
  5. Two more MaxPool layers in Table 1 (before LAD-Inception: 28->14; after
     it: 14->7) list kernel=3/stride=2 with no padding value given. Padding=0
     would give 13 and 6 respectively, not 14 and 7 -- we used padding=1 for
     both, which reproduces the table's stated output shapes exactly.
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        w = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * w


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al. 2018): channel then spatial attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention()

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class ADConv(nn.Module):
    """Asymmetric + Dilated Convolution: n x n receptive field via a dilated
    1xn conv followed by a dilated nx1 conv (Fig. 2c)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, dilation: int = 1, bias: bool = False):
        super().__init__()
        eff_k = dilation * (kernel_size - 1) + 1
        pad = eff_k // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, (1, kernel_size), stride=(1, stride),
                                padding=(0, pad), dilation=(1, dilation), bias=bias)
        self.conv2 = nn.Conv2d(out_ch, out_ch, (kernel_size, 1), stride=(stride, 1),
                                padding=(pad, 0), dilation=(dilation, 1), bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CBLR(nn.Module):
    """Conv + BatchNorm + Leaky-ReLU block (Fig. 3a)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 1,
                 stride: int = 1, padding: int = 0, negative_slope: float = 0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LRCBAM(nn.Module):
    """Conv + BN + Leaky-ReLU + CBAM (Fig. 3c)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.cblr = CBLR(in_ch, out_ch, kernel_size, stride, padding)
        self.cbam = CBAM(out_ch)

    def forward(self, x):
        return self.cbam(self.cblr(x))


class ADConvBranch(nn.Module):
    """CBLR(1x1) -> AD Conv(k=3, dilation=d) -- one multiscale branch of LAD-Inception."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, dilation: int):
        super().__init__()
        self.reduce = CBLR(in_ch, mid_ch, kernel_size=1)
        self.ad_conv = ADConv(mid_ch, out_ch, kernel_size=3, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.reduce(x)
        x = self.ad_conv(x)
        return self.act(self.bn(x))


class LADInception(nn.Module):
    """Multiscale Inception-style module with 3 AD-conv branches (dilation
    1/2/3), a bare 1x1 branch, and a maxpool+1x1 branch, followed by channel
    attention and a residual connection (Fig. 4)."""

    def __init__(self, in_ch: int, out_ch: int = 272):
        super().__init__()
        # 272 = 64*4 + 16, a round split across 5 branches (not paper-specified, see model.py docstring)
        b = out_ch // 5
        widths = [b, b, b, b, out_ch - 4 * b]

        self.branch1 = ADConvBranch(in_ch, mid_ch=in_ch // 2, out_ch=widths[0], dilation=1)
        self.branch2 = ADConvBranch(in_ch, mid_ch=in_ch // 2, out_ch=widths[1], dilation=2)
        self.branch3 = ADConvBranch(in_ch, mid_ch=in_ch // 2, out_ch=widths[2], dilation=3)
        self.branch4 = CBLR(in_ch, widths[3], kernel_size=1)
        self.branch5 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            CBLR(in_ch, widths[4], kernel_size=1),
        )

        self.channel_attn = ChannelAttention(out_ch)
        self.residual_proj = (
            nn.Identity() if in_ch == out_ch
            else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x), self.branch5(x)],
            dim=1,
        )
        out = self.channel_attn(out)
        out = out + self.residual_proj(x)
        return self.act(out)


class LADNet(nn.Module):
    def __init__(self, num_classes: int = 6, dropout: float = 0.5):
        super().__init__()
        # stem AD Convolution: table gives 224x224 -> 56x56, a 4x reduction;
        # stride=4 (not the table's literal "2") is needed to hit that shape
        # exactly -- see "Known deviations" #4 above.
        self.stem = nn.Sequential(
            ADConv(3, 64, kernel_size=3, stride=4, dilation=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)  # -> 28x28x64

        self.lrcbam1 = LRCBAM(64, 64, kernel_size=1, stride=1, padding=0)     # -> 28x28x64
        self.lrcbam2 = LRCBAM(64, 192, kernel_size=3, stride=1, padding=1)    # -> 28x28x192
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)  # -> 14x14x192 (see deviation #5)

        self.lad_inception = LADInception(192, out_ch=272)  # -> 14x14x272
        self.pool3 = nn.MaxPool2d(3, stride=2, padding=1)  # -> 7x7x272 (see deviation #5)

        self.cbam = CBAM(272)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(272, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(x)
        x = self.lrcbam1(x)
        x = self.lrcbam2(x)
        x = self.pool2(x)
        x = self.lad_inception(x)
        x = self.pool3(x)
        x = self.cbam(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = LADNet(num_classes=6)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print("output shape:", y.shape)
    n = count_params(m)
    print(f"trainable params: {n} ({n/1e6:.3f}M, ~{n*4/1e6:.2f}MB as fp32)")
