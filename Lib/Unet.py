# -*- coding = utf-8 -*-
# @time:2025/3/21 22:07
# Author:Zhihao.Xiong
# @File:Unet.py
# @Software:PyCharm
import torch

#  ------------------ U-Net ------------------
class _EncoderBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, polling=True, bn=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            torch.nn.BatchNorm2d(out_channels) if bn else torch.nn.GroupNorm(32, out_channels),
            torch.nn.GELU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            torch.nn.BatchNorm2d(out_channels) if bn else torch.nn.GroupNorm(32, out_channels),
            torch.nn.GELU(),
        ]
        if dropout:
            layers.append(torch.nn.Dropout())
        self.encode = torch.nn.Sequential(*layers)
        self.pool = None
        if polling:
            self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        return self.encode(x)


class _DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, bn=False):
        super(_DecoderBlock, self).__init__()
        self.decode = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            torch.nn.BatchNorm2d(middle_channels) if bn else torch.nn.GroupNorm(32, middle_channels),
            torch.nn.GELU(),
            torch.nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            torch.nn.BatchNorm2d(out_channels) if bn else torch.nn.GroupNorm(32, out_channels),
            torch.nn.GELU()
        )

    def forward(self, x):
        return self.decode(x)


class UNet4(torch.nn.Module):
    def __init__(self, num_classes, in_channels=3, bn=False, factors=2):
        super(UNet4, self).__init__()
        self.enc1 = _EncoderBlock(in_channels, 32 * factors, polling=False, bn=bn)
        self.enc2 = _EncoderBlock(32 * factors, 64 * factors, bn=bn)
        self.enc3 = _EncoderBlock(64 * factors, 128 * factors, bn=bn)
        self.enc4 = _EncoderBlock(128 * factors, 256 * factors, bn=bn)
        self.polling = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlock(256 * factors, 512 * factors, 256 * factors, bn=bn)

        self.dec4 = _DecoderBlock(512 * factors, 256 * factors, 128 * factors, bn=bn)
        self.dec3 = _DecoderBlock(256 * factors, 128 * factors, 64 * factors, bn=bn)
        self.dec2 = _DecoderBlock(128 * factors, 64 * factors, 32 * factors, bn=bn)
        self.dec1 = torch.nn.Sequential(
            torch.nn.Conv2d(64 * factors, 32 * factors, kernel_size=3, padding=1, padding_mode='reflect'),
            torch.nn.BatchNorm2d(32 * factors) if bn else torch.nn.GroupNorm(32, 32 * factors),
            torch.nn.GELU(),
            torch.nn.Conv2d(32 * factors, 32 * factors, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(32 * factors) if bn else torch.nn.GroupNorm(32, 32 * factors),
            torch.nn.GELU(),
        )
        self.final = torch.nn.Conv2d(32 * factors, num_classes, kernel_size=1)
        self.initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        dec4 = self.dec4(torch.cat([torch.nn.functional.interpolate(center, enc4.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc4], 1))
        dec3 = self.dec3(torch.cat([torch.nn.functional.interpolate(dec4, enc3.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc3], 1))
        dec2 = self.dec2(torch.cat([torch.nn.functional.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc2], 1))
        dec1 = self.dec1(torch.cat([torch.nn.functional.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc1], 1))
        final = self.final(dec1)
        return final

    def initialize_weights(*models):
        for model in models:
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, torch.nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

class UNet5(torch.nn.Module):
    def __init__(self, num_classes, in_channels=3, bn=False, channels=None):
        super(UNet5, self).__init__()
        if channels is None:
            channels = [32, 64, 128, 256, 512]
        self.channels = channels
        self.enc1 = _EncoderBlock(in_channels, channels[0], polling=False, bn=bn)
        self.enc2 = _EncoderBlock(channels[0], channels[1], bn=bn)
        self.enc3 = _EncoderBlock(channels[1], channels[2], bn=bn)
        self.enc4 = _EncoderBlock(channels[2], channels[3], bn=bn)
        self.enc5 = _EncoderBlock(channels[3], channels[4], bn=bn)
        self.polling = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlock(channels[4], channels[4]*2, channels[4], bn=bn)
        self.dec5 = _DecoderBlock(channels[4]*2, channels[3]*2, channels[3], bn=bn)
        self.dec4 = _DecoderBlock(channels[3]*2, channels[2]*2, channels[2], bn=bn)
        self.dec3 = _DecoderBlock(channels[2]*2, channels[1]*2, channels[1], bn=bn)
        self.dec2 = _DecoderBlock(channels[1]*2, channels[0]*2, channels[0], bn=bn)
        self.dec1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[0]*2, channels[0], kernel_size=3, padding=1, padding_mode='reflect'),
            torch.nn.BatchNorm2d(channels[0]) if bn else torch.nn.GroupNorm(channels[0], channels[0]),
            torch.nn.GELU(),
            torch.nn.Conv2d(channels[0], channels[0], kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(channels[0]) if bn else torch.nn.GroupNorm(channels[0], channels[0]),
            torch.nn.GELU(),
        )
        self.final = torch.nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        center = self.center(self.polling(enc5))
        dec5 = self.dec5(torch.cat([torch.nn.functional.interpolate(center, enc5.size()[-2:], align_corners=False,
                                                                    mode='bilinear'), enc5], 1))
        dec4 = self.dec4(torch.cat([torch.nn.functional.interpolate(dec5, enc4.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc4], 1))
        dec3 = self.dec3(torch.cat([torch.nn.functional.interpolate(dec4, enc3.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc3], 1))
        dec2 = self.dec2(torch.cat([torch.nn.functional.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc2], 1))
        dec1 = self.dec1(torch.cat([torch.nn.functional.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc1], 1))
        final = self.final(dec1)
        return final

    def initialize_weights(*models):
        for model in models:
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, torch.nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

#  ------------------ 3D U-Net ------------------
class UNet3D(torch.nn.Module):
    # __                            __
    #  1|__   ________________   __|1
    #     2|__  ____________  __|2
    #        3|__  ______  __|3
    #           4|__ __ __|4
    # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity

    def __init__(self, in_channels=32, out_channels=6, feat_channels=None, residual='conv'):

        #residual: 是否加入残差边，不加则为None
        super(UNet3D, self).__init__()

        # Encoder downsamplers
        if feat_channels is None:
            feat_channels = [32, 64, 64, 128, 128]

        self.pool1 = torch.nn.MaxPool3d((2,2,2))
        self.pool2 = torch.nn.MaxPool3d((2,2,2))
        self.pool3 = torch.nn.MaxPool3d((2,2,2))
        self.pool4 = torch.nn.MaxPool3d((2,2,2))

        # Encoder convolutions
        self.conv_blk1 = Conv3D_Block(in_channels, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual)
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual)
        self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual)
        self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual)

        # Decoder convolutions
        self.dec_conv_blk4 = Conv3D_Block(2*feat_channels[4], feat_channels[2], residual=residual)
        self.dec_conv_blk3 = Conv3D_Block(2*feat_channels[2], feat_channels[1], residual=residual)
        self.dec_conv_blk2 = Conv3D_Block(2*feat_channels[1], feat_channels[0], residual=residual)
        self.dec_conv_blk1 = Conv3D_Block(2*feat_channels[0], feat_channels[0], residual=residual)

        # Decoder upsamplers
        self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2],outpad=1)
        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

        # Final 1*1 Conv Segmentation map
        self.one_conv = torch.nn.Conv3d(feat_channels[0], out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        # Activation function
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):

        # 编码器
        x1 = self.conv_blk1(x)
        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)
        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)
        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)
        x_low4 = self.pool4(x4)
        base = self.conv_blk5(x_low4)


        # 解码器
        # d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
        # d_high4 = self.dec_conv_blk4(d4)
        # d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
        # d_high3 = self.dec_conv_blk3(d3)
        # d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        # d_high2 = self.dec_conv_blk2(d2)
        # d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        # d_high1 = self.dec_conv_blk1(d1)
        # seg = self.sigmoid(self.one_conv(d_high1))

        d4 = torch.cat([torch.nn.functional.interpolate(base, x4.size()[-3:], align_corners=False,
                                                                    mode='trilinear'), x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)
        d3 = torch.cat([torch.nn.functional.interpolate(d_high4, x3.size()[-3:], align_corners=False,
                                                                    mode='trilinear'), x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        d2 = torch.cat([torch.nn.functional.interpolate(d_high3, x2.size()[-3:], align_corners=False,
                                                                    mode='trilinear'), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        d1 = torch.cat([torch.nn.functional.interpolate(d_high2, x1.size()[-3:], align_corners=False,
                                                                    mode='trilinear'), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)
        seg = self.one_conv(d_high1)

        return seg


class Conv3D_Block(torch.nn.Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):

        super(Conv3D_Block, self).__init__()

        self.conv1 = torch.nn.Sequential(
                        torch.nn.Conv3d(inp_feat, out_feat, kernel_size=kernel,
                                    stride=stride, padding=padding, bias=True),
                        torch.nn.BatchNorm3d(out_feat),
                        torch.nn.ReLU())

        self.conv2 = torch.nn.Sequential(
                        torch.nn.Conv3d(out_feat, out_feat, kernel_size=kernel,
                                    stride=stride, padding=padding, bias=True),
                        torch.nn.BatchNorm3d(out_feat),
                        torch.nn.ReLU())

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = torch.nn.Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):

        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)


class Deconv3D_Block(torch.nn.Module):

    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1,outpad=0):

        super(Deconv3D_Block, self).__init__()

        self.deconv = torch.nn.Sequential(
                        #3D反卷积
                        torch.nn.ConvTranspose3d(inp_feat, out_feat, kernel_size=(kernel,kernel,kernel),
                                    stride=(stride,stride,stride), padding=(padding, padding, padding), output_padding=outpad, bias=True),
                        torch.nn.ReLU())

    def forward(self, x):

        return self.deconv(x)


class ChannelPool3d(torch.nn.AvgPool1d):

    def __init__(self, kernel_size, stride, padding):

        super(ChannelPool3d, self).__init__(kernel_size, stride, padding)
        self.pool_1d = torch.nn.AvgPool1d(self.kernel_size, self.stride, self.padding, self.ceil_mode)

    def forward(self, inp):
        n, c, d, w, h = inp.size()
        inp = inp.view(n,c,d*w*h).permute(0,2,1)
        pooled = self.pool_1d(inp)
        c = int(c/self.kernel_size[0])
        return inp.view(n,c,d,w,h)
