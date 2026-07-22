import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from torch.autograd import Variable
from einops import rearrange


def to_var(x, requires_grad=True):
    return Variable(x, requires_grad=requires_grad)


class MetaConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)

        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.groups = self.conv.groups
        self.kernel_size = self.conv.kernel_size
        self.weight = self.conv.weight
        self.bias = self.conv.bias

        self.weight_meta = to_var(self.conv.weight.data, requires_grad=True)
        if self.conv.bias is not None:
            self.bias_meta = to_var(self.conv.bias.data, requires_grad=True)
        else:
            self.bias_meta = None

    def named_leaves(self):
        return [("weight", self.weight), ("bias", self.bias)]

    def device_check(self, x):
        if self.weight_meta is not None and self.weight_meta.device != x.device:
            self.weight_meta = self.weight_meta.to(x.device)
        if self.bias_meta is not None and self.bias_meta.device != x.device:
            self.bias_meta = self.bias_meta.to(x.device)

    def forward(self, x, meta=False):
        if meta:
            self.device_check(x)
            return F.conv2d(
                x,
                self.weight_meta,
                self.bias_meta,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        else:
            return self.conv(x)


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

        self.weight_meta = to_var(self.weight.data, requires_grad=True)
        self.bias_meta = to_var(self.bias.data, requires_grad=True)

    def named_leaves(self):
        return [("weight", self.weight), ("bias", self.bias)]

    def device_check(self, x):
        if self.weight_meta.device != x.device:
            self.weight_meta = self.weight_meta.to(x.device)
        if self.bias_meta.device != x.device:
            self.bias_meta = self.bias_meta.to(x.device)

    def forward(self, x, meta=False):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        if meta:
            self.device_check(x)
            return (x - mu) / torch.sqrt(
                sigma + 1e-5
            ) * self.weight_meta + self.bias_meta
        else:
            return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class MetaSequential(nn.Sequential):
    def __init__(self, *args):
        super(MetaSequential, self).__init__(*args)

    def forward(self, x, meta=False):
        for module in self:
            if hasattr(module, "forward") and callable(module.forward):
                if hasattr(module, "supports_meta") and module.supports_meta:
                    x = module(x, meta=meta)
                else:
                    x = module(x)
            else:
                x = module(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x, meta=False):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x), meta=meta), h, w)


## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward_P(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_P, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = MetaConv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.project_out = MetaConv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.con1x1 = MetaConv2d(dim, hidden_features, kernel_size=1, bias=bias)

    def forward(self, x, p, meta=False):
        x = self.project_in(x, meta=meta)
        p = self.con1x1(p)
        x = F.gelu(x) * p
        x = self.project_out(x, meta=meta)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = MetaConv2d(
            in_channels=dim, out_channels=hidden_features * 2, kernel_size=1, bias=bias
        )
        self.dwconv = MetaConv2d(
            in_channels=hidden_features * 2,
            out_channels=hidden_features * 2,
            stride=1,
            padding=1,
            kernel_size=3,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = MetaConv2d(
            in_channels=hidden_features, out_channels=dim, kernel_size=1, bias=bias
        )

    def forward(self, x, meta=False):
        x = self.project_in(x, meta=meta)
        x1, x2 = self.dwconv(x, meta=meta).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x, meta=meta)
        return x


## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_meta = to_var(self.temperature.data)
        self.qkv = MetaConv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = MetaConv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = MetaConv2d(dim, dim, kernel_size=1, bias=bias)

    def named_leaves(self):
        return [("temperature", self.temperature)]

    def device_check(self, x):
        if (
            self.temperature_meta is not None
            and self.temperature_meta.device != x.device
        ):
            self.temperature_meta = self.temperature_meta.to(x.device)

    def forward(self, x, meta=False):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x, meta=meta), meta=meta)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        if meta:
            self.device_check(x)
            attn = (q @ k.transpose(-2, -1)) * self.temperature_meta
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out, meta=meta)
        return out


class TransformerBlock_P(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(TransformerBlock_P, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward_P(dim, ffn_expansion_factor, bias)

    def forward(self, x, p, meta=False):
        x = x + self.attn(self.norm1(x, meta=meta), meta=meta)
        p = p + self.attn(self.norm1(p, meta=meta), meta=meta)
        x = x + self.ffn(self.norm2(x, meta=meta), self.norm2(p, meta=meta), meta=meta)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, meta=False):
        x = x + self.attn(self.norm1(x, meta=meta), meta=meta)
        x = x + self.ffn(self.norm2(x, meta=meta), meta=meta)

        return x


# Feature Extractor Module
class ImgFeatureExtractorModule(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(ImgFeatureExtractorModule, self).__init__()

        self.proj = MetaSequential(
            MetaConv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.SiLU(),
        )

    def forward(self, x, meta=False):
        x = self.proj(x, meta=meta)

        return x


class PolarFeatureExtractorModule(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(PolarFeatureExtractorModule, self).__init__()

        self.proj = MetaSequential(
            MetaConv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.SiLU(),
        )

    def forward(self, x, meta=False):
        x = self.proj(x, meta=meta)

        return x


class Refinement(nn.Module):
    def __init__(self, dim):
        super(Refinement, self).__init__()

        self.refinement = MetaSequential(
            MetaConv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            MetaConv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            MetaConv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x, meta=False):
        return self.refinement(x, meta=meta)


class Refinement_down(nn.Module):
    def __init__(self, dim):
        super(Refinement_down, self).__init__()

        self.refinement = MetaSequential(
            MetaConv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            MetaConv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            MetaConv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            MetaConv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, meta=False):
        return self.refinement(x, meta=meta)


## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = MetaSequential(
            MetaConv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x, meta=False):
        return self.body(x, meta=meta)


class Downsample_conv(nn.Module):
    def __init__(self, n_feat):
        super(Downsample_conv, self).__init__()

        self.body = MetaConv2d(
            n_feat, n_feat * 2, kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, x, meta=False):
        return self.body(x, meta=meta)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = MetaSequential(
            MetaConv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x, meta=False):
        return self.body(x, meta=meta)


class Upsample_inter(nn.Module):
    def __init__(self, n_feat):
        super(Upsample_inter, self).__init__()

        self.body = MetaConv2d(
            n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x, meta=False):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.body(x, meta=meta)
        return x


class FeatureTransform(nn.Module):
    def __init__(self, dim):
        super(FeatureTransform, self).__init__()

        self.conv1 = MetaSequential(
            MetaConv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
            LayerNorm(dim * 2),
            nn.SiLU(),
        )
        self.conv2 = MetaSequential(
            MetaConv2d(
                dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            LayerNorm(dim * 2),
            nn.SiLU(),
        )
        self.conv3 = MetaConv2d(
            dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x, meta=False):
        x = self.conv1(x, meta=meta)
        x = self.conv2(x, meta=meta)
        x = self.conv3(x, meta=meta)

        return x


class FT(nn.Module):
    def __init__(self, dim=48):
        super(FT, self).__init__()

        self.FT1 = FeatureTransform(dim * 2)
        self.FT2 = FeatureTransform(dim * 2)
        self.FT3 = FeatureTransform(dim * 4)

    def forward(self, demfeat, meta=False):
        TF = {}
        TF["TF1"] = F.interpolate(
            self.FT1(demfeat["DF1"], meta=meta), scale_factor=2, mode="bilinear"
        )
        TF["TF2"] = F.interpolate(
            self.FT2(demfeat["DF2"], meta=meta), scale_factor=2, mode="bilinear"
        )
        TF["TF3"] = F.interpolate(
            self.FT3(demfeat["DF3"], meta=meta), scale_factor=2, mode="bilinear"
        )

        return TF


class MetaFeatureGenerator(nn.Module):
    def __init__(self, dim):
        super(MetaFeatureGenerator, self).__init__()

        self.conv_task = MetaSequential(
            MetaConv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
            LayerNorm(dim * 2),
            nn.SiLU(),
            MetaConv2d(
                dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            LayerNorm(dim * 2),
            nn.SiLU(),
            MetaConv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, xd, xt, meta=False):
        xt = self.conv_task(xt, meta=meta)
        return xt


class MFG(nn.Module):
    def __init__(self, dim=48):
        super(MFG, self).__init__()

        self.MFG1 = MetaFeatureGenerator(dim * 2)
        self.MFG2 = MetaFeatureGenerator(dim * 2)
        self.MFG3 = MetaFeatureGenerator(dim * 4)

    def forward(self, demfeat, taskfeat, meta=False):
        MF = {}
        MF["MF1"] = self.MFG1(demfeat["DF1"], taskfeat["TaF1"], meta=meta)
        MF["MF2"] = self.MFG2(demfeat["DF2"], taskfeat["TaF2"], meta=meta)
        MF["MF3"] = self.MFG3(demfeat["DF3"], taskfeat["TaF3"], meta=meta)

        return MF


class FeatureAlignment(nn.Module):
    def __init__(self):
        super(FeatureAlignment, self).__init__()

        self.FT = FT()
        self.MFG = MFG()

    def forward(self, demfeat, taskfeat, meta=False):
        x1 = self.FT(demfeat, meta=meta)
        x2 = self.MFG(demfeat, taskfeat, meta=meta)

        return x1, x2


class PIDNet(nn.Module):
    def __init__(
        self,
        inp_channels=12,
        out_channels=12,
        dim=48,
        num_blocks=[4, 4, 4],
        heads=[1, 2, 4],
        ffn_expansion_factor=2.66,
        bias=False,
    ):
        super(PIDNet, self).__init__()

        # Feature Extractor
        self.img_feature_extractor = ImgFeatureExtractorModule(inp_channels, dim)
        self.polar_feature_extractor = PolarFeatureExtractorModule(6, dim)

        # ------------------------ Encoder ------------------------#
        # Stage 1
        self.encoder1 = MetaSequential(
            *[
                TransformerBlock_P(dim, heads[0], ffn_expansion_factor, bias)
                for _ in range(num_blocks[0])
            ]
        )
        self.down1 = Downsample_conv(dim)

        # Stage 2
        self.encoder2 = MetaSequential(
            *[
                TransformerBlock_P(dim * 2, heads[1], ffn_expansion_factor, bias)
                for _ in range(num_blocks[1])
            ]
        )
        self.down2 = Downsample_conv(dim * 2)

        # ------------------------ Bottleneck ------------------------#
        self.bottleneck = MetaSequential(
            *[
                TransformerBlock_P(dim * 4, heads[2], ffn_expansion_factor, bias)
                for _ in range(num_blocks[2])
            ]
        )

        # ------------------------ Decoder ------------------------#
        # Stage 3 -> 2
        self.up3 = Upsample_inter(dim * 4)
        self.reduce_chan3 = MetaConv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder2 = MetaSequential(
            *[
                TransformerBlock_P(dim * 2, heads[1], ffn_expansion_factor, bias)
                for _ in range(num_blocks[1])
            ]
        )

        # Stage 2 -> 1
        self.up2 = Upsample_inter(dim * 2)
        self.decoder1 = MetaSequential(
            *[
                TransformerBlock_P(dim * 2, heads[0], ffn_expansion_factor, bias)
                for _ in range(num_blocks[0])
            ]
        )

        self.con1x1 = MetaConv2d(dim, dim * 2, kernel_size=1, bias=bias)

        # ------------------------ Outputs ------------------------#
        self.upsample = Upsample_inter(dim * 2)

        self.refinement = Refinement(dim)
        self.output = MetaConv2d(dim, out_channels, 3, 1, 1, bias=bias)

    def calculate_polar(self, img):
        I0, I45, I90, I135 = img[:, 0:3], img[:, 3:6], img[:, 6:9], img[:, 9:12]
        S1 = (I0 - I90 + 1) / 2
        S2 = (I45 - I135 + 1) / 2
        return torch.cat([S1, S2], 1)

    def forward(self, inp_img, ELT_state=True, meta=False):
        polar = self.calculate_polar(inp_img)
        polar_feat = self.polar_feature_extractor(polar, meta=meta)  # (B, C, H, W)

        # ------------------------ Encoder ------------------------#
        img_feat = self.img_feature_extractor(inp_img, meta=meta)  # (B, C, H, W)

        # Encoder 1
        x1 = self.encoder1[0](img_feat, polar_feat, meta=meta)  # (B, C, H, W)
        for block in self.encoder1[1:]:
            x1 = block(x1, polar_feat, meta=meta)
        x1_down = self.down1(x1, meta=meta)  # (B, 2C, H/2, W/2)
        p1_down = self.down1(polar_feat, meta=meta)  # (B, 2C, H/2, W/2)

        # Encoder 2
        x2 = self.encoder2[0](x1_down, p1_down, meta=meta)  # (B, 2C, H/2, W/2)
        for block in self.encoder2[1:]:
            x2 = block(x2, p1_down, meta=meta)
        x2_down = self.down2(x2, meta=meta)  # (B, 4C, H/4, W/4)
        p2_down = self.down2(p1_down, meta=meta)  # (B, 4C, H/4, W/4)

        # ------------------------ Bottleneck ------------------------#
        x3 = self.bottleneck[0](x2_down, p2_down, meta=meta)  # (B, 4C, H/4, W/4)
        for block in self.bottleneck[1:]:
            x3 = block(x3, p2_down, meta=meta)

        # ------------------------ Decoder ------------------------#
        # Stage 3 -> 2
        x3_up = self.up3(x3, meta=meta)  # (B, 2C, H/2, W/2)
        x3_up = self.reduce_chan3(
            torch.cat([x3_up, x2], 1), meta=meta
        )  # (B, 2C, H/2, W/2)
        x2_dec = self.decoder2[0](x3_up, p1_down, meta=meta)  # (B, 2C, H/2, W/2)
        for block in self.decoder2[1:]:
            x2_dec = block(x2_dec, p1_down, meta=meta)

        # Stage 2 -> 1
        x2_up = self.up2(x2_dec, meta=meta)  # (B, C, H, W)
        x2_up = torch.cat([x2_up, x1], 1)  # (B, 2C, H, W)
        polar_feat = self.con1x1(polar_feat, meta=meta)
        x1_dec = self.decoder1[0](x2_up, polar_feat, meta=meta)  # (B, 2C, H, W)
        for block in self.decoder1[1:]:
            x1_dec = block(x1_dec, polar_feat, meta=meta)

        imgout = self.output(
            self.refinement(self.upsample(x1_dec, meta=meta), meta=meta), meta=meta
        ) + F.interpolate(inp_img, scale_factor=2, mode="bilinear")

        if ELT_state:
            return imgout
        else:
            demfeat = {}
            demfeat["DF3"] = x3
            demfeat["DF2"] = x2_dec
            demfeat["DF1"] = x1_dec
            return imgout, demfeat


class SfPNet(nn.Module):
    def __init__(
        self,
        inp_channels=7,
        out_channels=3,
        dim=48,
        num_blocks=[4, 4, 4],
        heads=[1, 2, 4],
        ffn_expansion_factor=2.66,
        bias=False,
    ):
        super(SfPNet, self).__init__()

        # Feature Extractor
        self.img_feature_extractor = ImgFeatureExtractorModule(inp_channels, dim)

        # ------------------------ Encoder ------------------------#
        # Stage 1
        self.encoder1 = MetaSequential(
            *[
                TransformerBlock(dim, heads[0], ffn_expansion_factor, bias)
                for _ in range(num_blocks[0])
            ]
        )
        self.down1 = Downsample_conv(dim)

        # Stage 2
        self.encoder2 = MetaSequential(
            *[
                TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias)
                for _ in range(num_blocks[1])
            ]
        )
        self.down2 = Downsample_conv(dim * 2)

        # ------------------------ Bottleneck ------------------------#
        self.bottleneck = MetaSequential(
            *[
                TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias)
                for _ in range(num_blocks[2])
            ]
        )

        # -------------------------- Decoder -------------------------#
        # Stage 3 -> 2
        self.up3 = Upsample_inter(dim * 4)
        self.reduce_chan3 = MetaConv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder2 = MetaSequential(
            *[
                TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias)
                for _ in range(num_blocks[1])
            ]
        )

        # Stage 2 -> 1
        self.up2 = Upsample_inter(dim * 2)
        self.decoder1 = MetaSequential(
            *[
                TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias)
                for _ in range(num_blocks[0])
            ]
        )

        # ------------------------ Outputs ------------------------#
        self.refinement = Refinement_down(dim * 2)
        self.output = MetaConv2d(dim, out_channels, 3, 1, 1, bias=bias)

    def forward(self, inp_img, meta=False):
        # ------------------------ Encoder ------------------------#
        img_feat = self.img_feature_extractor(inp_img, meta=meta)  # (B, C, H, W)

        # Encoder 1
        x1 = self.encoder1[0](img_feat, meta=meta)  # (B, C, H, W)
        for block in self.encoder1[1:]:
            x1 = block(x1, meta=meta)
        x1_down = self.down1(x1, meta=meta)  # (B, 2C, H/2, W/2)

        # Encoder 2
        x2 = self.encoder2[0](x1_down, meta=meta)  # (B, 2C, H/2, W/2)
        for block in self.encoder2[1:]:
            x2 = block(x2, meta=meta)
        x2_down = self.down2(x2, meta=meta)  # (B, 4C, H/4, W/4)

        # ------------------------ Bottleneck ---------------------#
        x3 = self.bottleneck[0](x2_down, meta=meta)  # (B, 4C, H/4, W/4)
        for block in self.bottleneck[1:]:
            x3 = block(x3, meta=meta)

        # ------------------------ Decoder ------------------------#
        # Stage 3 -> 2
        x3_up = self.up3(x3, meta=meta)  # (B, 2C, H/2, W/2)
        x3_up = self.reduce_chan3(
            torch.cat([x3_up, x2], 1), meta=meta
        )  # (B, 2C, H/2, W/2)
        x2_dec = self.decoder2[0](x3_up, meta=meta)  # (B, 2C, H/2, W/2)
        for block in self.decoder2[1:]:
            x2_dec = block(x2_dec, meta=meta)

        # Stage 2 -> 1
        x2_up = self.up2(x2_dec, meta=meta)  # (B, C, H, W)
        x2_up = torch.cat([x2_up, x1], 1)  # (B, 2C, H, W)
        x1_dec = self.decoder1[0](x2_up, meta=meta)  # (B, 2C, H, W)
        for block in self.decoder1[1:]:
            x1_dec = block(x1_dec, meta=meta)

        imgout = self.output(self.refinement(x1_dec, meta=meta), meta=meta)

        taskfeat = {}
        taskfeat["TaF3"] = x3
        taskfeat["TaF2"] = x2_dec
        taskfeat["TaF1"] = x1_dec

        return imgout, taskfeat


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PIDNet().to(device)
    inp = torch.randn(1, 12, 512, 512).to(device)
    out, out1 = model(inp)
    print(out.size(), out1.size())
