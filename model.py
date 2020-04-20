from torch import nn, cat, sum, tensor
from torch.nn.functional import affine_grid, grid_sample
from torchvision.utils import save_image
import torch


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ReplicationPad2d(padding=1),
        nn.Conv2d(in_channels, out_channels, 3, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.ReplicationPad2d(padding=1),
        nn.Conv2d(out_channels, out_channels, 3, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.ReplicationPad2d(padding=1),
        nn.Conv2d(out_channels, out_channels, 3, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
    )


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.pad = nn.ReplicationPad2d(padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        x = self.leaky_relu(x)

        res = self.pad(x)
        res = self.conv2(res)
        res = self.leaky_relu(res)

        res = self.pad(res)
        res = self.conv3(res)
        res = self.leaky_relu(res)

        out = res + x

        return out


class CABlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, reduction=16):
        super(CABlock, self).__init__()

        self.conv_block = conv_block(in_channels, out_channels)

        self.attetion = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block(x)
        y = self.attetion(x)
        out = x * y

        return out


class RCABlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, reduction=16):
        super(RCABlock, self).__init__()

        self.conv_block = nn.Sequential(
            # nn.ReplicationPad2d(padding=1),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.ReplicationPad2d(padding=1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.ReplicationPad2d(padding=1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.attetion = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.conv_block(x)
        y = self.attetion(features)
        out = features * y
        out = out + x

        return out


class ResUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(ResUNet, self).__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down1 = ResBlock(in_channels, 32)
        self.down2 = ResBlock(32, 64)
        self.down3 = ResBlock(64, 128)
        self.down4 = ResBlock(128, 256)

        self.inter_conv = ResBlock(256, 512)

        self.up4 = ResBlock(512 + 256, 256)
        self.up3 = ResBlock(256 + 128, 128)
        self.up2 = ResBlock(128 + 64, 64)
        self.up1 = ResBlock(64 + 32, 32)

        self.last_conv = nn.Sequential(nn.ReplicationPad2d(padding=1),
                                       nn.Conv2d(32, out_channels, 3))

    def forward(self, x):

        conv1 = self.down1(x)
        x = self.maxpool(conv1)

        conv2 = self.down2(x)
        x = self.maxpool(conv2)

        conv3 = self.down3(x)
        x = self.maxpool(conv3)

        conv4 = self.down4(x)
        x = self.maxpool(conv4)

        low_features = self.inter_conv(x)

        x = self.upsample(low_features)
        x = cat([x, conv4], dim=1)
        x = self.up4(x)

        x = self.upsample(x)
        x = cat([x, conv3], dim=1)
        x = self.up3(x)

        x = self.upsample(x)
        x = cat([x, conv2], dim=1)
        x = self.up2(x)

        x = self.upsample(x)
        x = cat([x, conv1], dim=1)
        x = self.up1(x)

        out = self.last_conv(x)

        return out


class CAUNet_res_image(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(CAUNet_res_image, self).__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.localize = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(512, 512, 3, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(512, 512, 3, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2)
        )

        self.regress = nn.Conv2d(512, 6, kernel_size=7)

        self.down1 = conv_block(in_channels, 32)
        self.down2 = conv_block(32, 64)
        self.down3 = conv_block(64, 128)
        self.down4 = conv_block(128, 256)

        self.inter_conv = CABlock(256, 512)

        self.up4 = CABlock(512 + 256, 256)
        self.up3 = CABlock(256 + 128, 128)
        self.up2 = CABlock(128 + 64, 64)
        self.up1 = CABlock(64 + 32, 32)

        self.last_conv = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(32, out_channels, 3, padding=0))

    def forward(self, x_in):

        conv1 = self.down1(x_in)
        x = self.maxpool(conv1)

        conv2 = self.down2(x)
        x = self.maxpool(conv2)

        conv3 = self.down3(x)
        x = self.maxpool(conv3)

        conv4 = self.down4(x)
        x = self.maxpool(conv4)

        low_features = self.inter_conv(x)

        x = self.upsample(low_features)
        x = cat([x, conv4], dim=1)
        x = self.up4(x)

        x = self.upsample(x)
        x = cat([x, conv3], dim=1)
        x = self.up3(x)

        x = self.upsample(x)
        x = cat([x, conv2], dim=1)
        x = self.up2(x)

        x = self.upsample(x)
        x = cat([x, conv1], dim=1)
        x = self.up1(x)

        out = self.last_conv(x)

        out = out + x_in

        return out


class CAUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(CAUNet, self).__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.localize = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(512, 512, 3, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(512, 512, 3, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2)
        )

        self.regress = nn.Conv2d(512, 6, kernel_size=7)

        self.down1 = conv_block(in_channels, 32)
        self.down2 = conv_block(32, 64)
        self.down3 = conv_block(64, 128)
        self.down4 = conv_block(128, 256)

        self.inter_conv = CABlock(256, 512)

        self.up4 = CABlock(512 + 256, 256)
        self.up3 = CABlock(256 + 128, 128)
        self.up2 = CABlock(128 + 64, 64)
        self.up1 = CABlock(64 + 32, 32)

        self.last_conv = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(32, out_channels, 3, padding=0))

    def forward(self, x):

        conv1 = self.down1(x)
        x = self.maxpool(conv1)

        conv2 = self.down2(x)
        x = self.maxpool(conv2)

        conv3 = self.down3(x)
        x = self.maxpool(conv3)

        conv4 = self.down4(x)
        x = self.maxpool(conv4)

        low_features = self.inter_conv(x)

        x = self.upsample(low_features)
        x = cat([x, conv4], dim=1)
        x = self.up4(x)

        x = self.upsample(x)
        x = cat([x, conv3], dim=1)
        x = self.up3(x)

        x = self.upsample(x)
        x = cat([x, conv2], dim=1)
        x = self.up2(x)

        x = self.upsample(x)
        x = cat([x, conv1], dim=1)
        x = self.up1(x)

        x = x - conv1

        out = self.last_conv(x)

        return out


class CAUNet_(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(CAUNet_, self).__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down1 = conv_block(in_channels, 32)
        self.down2 = conv_block(32, 64)
        self.down3 = conv_block(64, 128)
        self.down4 = conv_block(128, 256)

        self.inter_conv = CABlock(256, 512)

        self.up4 = CABlock(512 + 256, 256)
        self.up3 = CABlock(256 + 128, 128)
        self.up2 = CABlock(128 + 64, 64)
        self.up1 = CABlock(64 + 32, 32)

        self.last_conv = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(32, out_channels, 3, padding=0))

    def forward(self, x):

        conv1 = self.down1(x)
        x = self.maxpool(conv1)

        conv2 = self.down2(x)
        x = self.maxpool(conv2)

        conv3 = self.down3(x)
        x = self.maxpool(conv3)

        conv4 = self.down4(x)
        x = self.maxpool(conv4)

        low_features = self.inter_conv(x)

        x = self.upsample(low_features)
        x = cat([x, conv4], dim=1)
        x = self.up4(x)

        x = self.upsample(x)
        x = cat([x, conv3], dim=1)
        x = self.up3(x)

        x = self.upsample(x)
        x = cat([x, conv2], dim=1)
        x = self.up2(x)

        x = self.upsample(x)
        x = cat([x, conv1], dim=1)
        x = self.up1(x)

        x = x + conv1

        out = self.last_conv(x)

        return out


class CAUNet_DCN(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, im_size=224):
        super(CAUNet_DCN, self).__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down1 = conv_block(in_channels, 32)
        self.down2 = conv_block(32, 64)
        self.down3 = conv_block(64, 128)
        self.down4 = conv_block(128, 256)

        self.inter_conv = CABlock(256, 512)

        self.up4 = CABlock(512 + 256, 256)
        self.up3 = CABlock(256 + 128, 128)
        self.up2 = CABlock(128 + 64, 64)
        self.up1 = CABlock(64 + 32, 32)

        self.last_conv = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(32, out_channels, 3, padding=0))

        self.repli_pad = nn.ReplicationPad2d(1)
        self.offset = nn.Conv2d(32, 2, 3, padding=0)

        self.offset.weight.data.zero_()
        self.offset.bias.data.zero_()

        regular_grid = -torch.ones([1, im_size, im_size, 2], dtype=torch.float).cuda()

        for i in range(224):
            for j in range(224):
                regular_grid[0][i][j][0] += 2 * j / (im_size - 1)
                regular_grid[0][i][j][1] += 2 * i / (im_size - 1)

        self.regular_grid = regular_grid.detach()

    def forward(self, x):

        conv1 = self.down1(x)
        x = self.maxpool(conv1)

        conv2 = self.down2(x)
        x = self.maxpool(conv2)

        conv3 = self.down3(x)
        x = self.maxpool(conv3)

        conv4 = self.down4(x)
        x = self.maxpool(conv4)

        low_features = self.inter_conv(x)

        x = self.upsample(low_features)
        x = cat([x, conv4], dim=1)
        x = self.up4(x)

        x = self.upsample(x)
        x = cat([x, conv3], dim=1)
        x = self.up3(x)

        x = self.upsample(x)
        x = cat([x, conv2], dim=1)
        x = self.up2(x)

        x = self.upsample(x)
        x = cat([x, conv1], dim=1)
        x = self.up1(x)

        out_init = self.last_conv(x)

        offset = self.repli_pad(x)
        offset = self.offset(offset)
        offset = offset.permute([0, 2, 3, 1])
        grid = self.regular_grid + offset

        out = grid_sample(out_init, grid, padding_mode='border')
        return out, out_init


class DCN(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, im_size=224):
        super(DCN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ReplicationPad2d(1),
        )

        self.regress = nn.Conv2d(32, 2, 3, padding=0)
        self.regress.weight.data.zero_()
        self.regress.bias.data.zero_()

        # regular grid [-1, 1]
        regular_grid = -torch.ones([1, im_size, im_size, 2], dtype=torch.float).cuda()

        for i in range(224):
            for j in range(224):
                regular_grid[0][i][j][0] += 2 * j / (im_size - 1)
                regular_grid[0][i][j][1] += 2 * i / (im_size - 1)

        self.regular_grid = regular_grid.detach()

    def forward(self, x):
        offset = self.conv_block(x)
        offset = self.regress(offset)
        offset = offset.permute([0, 2, 3, 1])
        grid = self.regular_grid + offset
        out = grid_sample(x, grid, padding_mode='border')
        return out


class CAUNet_res_DCN(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, im_size=224):
        super(CAUNet_res_DCN, self).__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)



        self.down1 = conv_block(in_channels, 32)
        self.down2 = conv_block(32, 64)
        self.down3 = conv_block(64, 128)
        self.down4 = conv_block(128, 256)

        self.inter_conv = CABlock(256, 512)

        self.up4 = CABlock(512 + 256, 256)
        self.up3 = CABlock(256 + 128, 128)
        self.up2 = CABlock(128 + 64, 64)
        self.up1 = CABlock(64 + 32, 32)

        self.last_conv = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(32, out_channels, 3, padding=0))

        # self.dcn1 = DCN(32, 32)

    def forward(self, x):

        conv1 = self.down1(x)
        x = self.maxpool(conv1)

        conv2 = self.down2(x)
        x = self.maxpool(conv2)

        conv3 = self.down3(x)
        x = self.maxpool(conv3)

        conv4 = self.down4(x)
        x = self.maxpool(conv4)

        low_features = self.inter_conv(x)

        x = self.upsample(low_features)
        x = cat([x, conv4], dim=1)
        x = self.up4(x)

        x = self.upsample(x)
        x = cat([x, conv3], dim=1)
        x = self.up3(x)

        x = self.upsample(x)
        x = cat([x, conv2], dim=1)
        x = self.up2(x)

        x = self.upsample(x)
        x = cat([x, conv1], dim=1)
        x = self.up1(x)

        y = x - conv1

        # y = self.dcn1(y)

        out = self.last_conv(y)

        return out


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(UNet, self).__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down1 = conv_block(in_channels, 32)
        self.down2 = conv_block(32, 64)
        self.down3 = conv_block(64, 128)
        self.down4 = conv_block(128, 256)

        self.inter_conv = conv_block(256, 512)

        self.up4 = conv_block(512 + 256, 256)
        self.up3 = conv_block(256 + 128, 128)
        self.up2 = conv_block(128 + 64, 64)
        self.up1 = conv_block(64 + 32, 32)

        self.last_conv = nn.Sequential(nn.ReplicationPad2d(padding=1),
                                       nn.Conv2d(32, out_channels, 3))

    def forward(self, x):

        conv1 = self.down1(x)
        x = self.maxpool(conv1)

        conv2 = self.down2(x)
        x = self.maxpool(conv2)

        conv3 = self.down3(x)
        x = self.maxpool(conv3)

        conv4 = self.down4(x)
        x = self.maxpool(conv4)

        low_features = self.inter_conv(x)

        x = self.upsample(low_features)
        x = cat([x, conv4], dim=1)
        x = self.up4(x)

        x = self.upsample(x)
        x = cat([x, conv3], dim=1)
        x = self.up3(x)

        x = self.upsample(x)
        x = cat([x, conv2], dim=1)
        x = self.up2(x)

        x = self.upsample(x)
        x = cat([x, conv1], dim=1)
        x = self.up1(x)

        out = self.last_conv(x)

        return out, low_features


class BasicConv(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3):
        super(BasicConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.conv_block(x)


class STN(nn.Module):
    def __init__(self, in_channels=3):
        super(STN, self).__init__()

        self.conv1 = BasicConv(in_channels, 32)
        self.conv2 = BasicConv(32, 64)
        self.conv3 = BasicConv(64, 128)
        self.conv4 = BasicConv(128, 256)
        self.conv5 = BasicConv(256, 512)

        self.conv_last = nn.Conv2d(512, 6, kernel_size=7)

        self.conv_last.weight.data.zero_()
        self.conv_last.bias.data.copy_(tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):

        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)
        tmp = self.conv4(tmp)
        tmp = self.conv5(tmp)

        theta = self.conv_last(tmp)
        theta = theta.view(-1, 2, 3)

        grid = affine_grid(theta, x.size())
        out = grid_sample(x, grid)

        return out


class AffineTransform(nn.Module):
    def __init__(self):
        super(AffineTransform, self).__init__()

    def forward(self, theta, x, padding_mode='zeros'):
        theta = theta.view(-1, 2, 3)
        grid = affine_grid(theta, x.size())
        out = grid_sample(x, grid, padding_mode=padding_mode)
        return out


class CABlock_STN(nn.Module):
    def __init__(self, in_channels=32, out_channels=32):
        super(CABlock_STN, self).__init__()

        self.conv_block = CABlock(in_channels, out_channels)

        self.stn = STN(out_channels)

    def forward(self, x):
        x = self.conv_block(x)
        out = self.stn(x)
        return out


class UNet_STN(nn.Module):
    def __init__(self):
        super(UNet_STN, self).__init__()

        self.base = CAUNet()
        self.stn = STN()

    def forward(self, x):
        rgb_init = self.base(x)
        rgb_trans = self.stn(rgb_init)

        return rgb_init, rgb_trans


class STN_1(nn.Module):
    def __init__(self, in_channels=32):
        super(STN_1, self).__init__()

        self.conv1 = BasicConv(in_channels, 32)
        self.conv2 = BasicConv(32, 64)
        self.conv3 = BasicConv(64, 128)
        self.conv4 = BasicConv(128, 256)
        self.conv5 = BasicConv(256, 512)

        self.regress = nn.Conv2d(512, 6, kernel_size=7)
        self.regress.weight.data.zero_()
        self.regress.bias.data.copy_(tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.affine = AffineTransform()

    def forward(self, x):
        loc = self.conv1(x)
        loc = self.conv2(loc)
        loc = self.conv3(loc)
        loc = self.conv4(loc)
        loc = self.conv5(loc)
        theta = self.regress(loc)
        out = self.affine(theta, x)

        return out


class STN_2(nn.Module):
    def __init__(self, in_channels=64):
        super(STN_2, self).__init__()

        self.conv1 = BasicConv(in_channels, 64)
        self.conv2 = BasicConv(64, 128)
        self.conv3 = BasicConv(128, 256)
        self.conv4 = BasicConv(256, 512)

        self.regress = nn.Conv2d(512, 6, kernel_size=7)
        self.regress.weight.data.zero_()
        self.regress.bias.data.copy_(tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.affine = AffineTransform()

    def forward(self, x):
        loc = self.conv1(x)
        loc = self.conv2(loc)
        loc = self.conv3(loc)
        loc = self.conv4(loc)
        theta = self.regress(loc)
        out = self.affine(theta, x)

        return out


class STN_3(nn.Module):
    def __init__(self, in_channels=128):
        super(STN_3, self).__init__()

        self.conv1 = BasicConv(in_channels, 128)
        self.conv2 = BasicConv(128, 256)
        self.conv3 = BasicConv(256, 512)

        self.regress = nn.Conv2d(512, 6, kernel_size=7)
        self.regress.weight.data.zero_()
        self.regress.bias.data.copy_(tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.affine = AffineTransform()

    def forward(self, x):
        loc = self.conv1(x)
        loc = self.conv2(loc)
        loc = self.conv3(loc)
        theta = self.regress(loc)
        out = self.affine(theta, x)

        return out


class STN_4(nn.Module):
    def __init__(self, in_channels=256):
        super(STN_4, self).__init__()

        self.conv1 = BasicConv(in_channels, 256)
        self.conv2 = BasicConv(256, 512)

        self.regress = nn.Conv2d(512, 6, kernel_size=7)
        self.regress.weight.data.zero_()
        self.regress.bias.data.copy_(tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.affine = AffineTransform()

    def forward(self, x):
        loc = self.conv1(x)
        loc = self.conv2(loc)
        theta = self.regress(loc)
        out = self.affine(theta, x)

        return out


class STN_5(nn.Module):
    def __init__(self, in_channels=512):
        super(STN_5, self).__init__()

        self.localize = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, 512, 3, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(512, 512, 3, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2)
        )

        self.regress = nn.Conv2d(512, 6, kernel_size=7)
        self.regress.weight.data.zero_()
        self.regress.bias.data.copy_(tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.affine = AffineTransform()

    def forward(self, x):
        loc = self.localize(x)
        theta = self.regress(loc)
        out = self.affine(theta, x)

        return out


class STN_CAUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(STN_CAUNet, self).__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.affine = AffineTransform()
        self.localize = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(512, 512, 3, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(512, 512, 3, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2)
        )

        self.regress = nn.Conv2d(512, 6, kernel_size=7)
        self.regress.weight.data.zero_()
        self.regress.bias.data.copy_(tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.down1 = conv_block(in_channels, 32)
        self.down2 = conv_block(32, 64)
        self.down3 = conv_block(64, 128)
        self.down4 = conv_block(128, 256)

        self.inter_conv = CABlock(256, 512)

        self.up4 = CABlock(512 + 256, 256)
        self.up3 = CABlock(256 + 128, 128)
        self.up2 = CABlock(128 + 64, 64)
        self.up1 = CABlock(64 + 32, 32)

        self.last_conv = nn.Sequential(nn.ReplicationPad2d(padding=1),
                                       nn.Conv2d(32, out_channels, 3))

    def forward(self, x):

        conv1 = self.down1(x)
        x = self.maxpool(conv1)

        conv2 = self.down2(x)
        x = self.maxpool(conv2)

        conv3 = self.down3(x)
        x = self.maxpool(conv3)

        conv4 = self.down4(x)
        x = self.maxpool(conv4)

        x = self.inter_conv(x)

        loc = self.localize(x)
        theta = self.regress(loc)
        x = self.affine(theta, x)

        x = self.upsample(x)
        x = cat([x, conv4], dim=1)
        x = self.up4(x)

        x = self.upsample(x)
        x = cat([x, conv3], dim=1)
        x = self.up3(x)

        x = self.upsample(x)
        x = cat([x, conv2], dim=1)
        x = self.up2(x)

        x = self.upsample(x)
        x = cat([x, conv1], dim=1)
        x = self.up1(x)

        out = self.last_conv(x)

        return out


class STN_all_CAUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(STN_all_CAUNet, self).__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.stn1 = STN_1(32)
        self.stn2 = STN_2(64)
        self.stn3 = STN_3(128)
        self.stn4 = STN_4(256)
        self.stn5 = STN_5(512)

        self.down1 = conv_block(in_channels, 32)
        self.down2 = conv_block(32, 64)
        self.down3 = conv_block(64, 128)
        self.down4 = conv_block(128, 256)

        self.inter_conv = CABlock(256, 512)

        self.up4 = CABlock(512 + 256, 256)
        self.up3 = CABlock(256 + 128, 128)
        self.up2 = CABlock(128 + 64, 64)
        self.up1 = CABlock(64 + 32, 32)

        self.last_conv = nn.Sequential(nn.ReplicationPad2d(padding=1),
                                       nn.Conv2d(32, out_channels, 3))

    def forward(self, x):
        conv1 = self.down1(x)
        conv1_trans = self.stn1(conv1)
        x = self.maxpool(conv1)

        conv2 = self.down2(x)
        conv2_trans = self.stn2(conv2)
        x = self.maxpool(conv2)

        conv3 = self.down3(x)
        conv3_trans = self.stn3(conv3)
        x = self.maxpool(conv3)

        conv4 = self.down4(x)
        conv4_trans = self.stn4(conv4)
        x = self.maxpool(conv4)

        x = self.inter_conv(x)

        x = self.stn5(x)

        x = self.upsample(x)
        x = cat([x, conv4_trans], dim=1)
        x = self.up4(x)

        x = self.upsample(x)
        x = cat([x, conv3_trans], dim=1)
        x = self.up3(x)

        x = self.upsample(x)
        x = cat([x, conv2_trans], dim=1)
        x = self.up2(x)

        x = self.upsample(x)
        x = cat([x, conv1_trans], dim=1)
        x = self.up1(x)

        out = self.last_conv(x)

        return out


class GlobalColorCorrector(nn.Module):
    def __init__(self, in_channels=512):
        super(GlobalColorCorrector, self).__init__()

        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = conv_block(512, 512)
        self.conv2 = nn.Conv2d(in_channels, 1024, kernel_size=3, stride=2)
        self.conv_r = nn.Conv2d(1024, 3, kernel_size=3, stride=2)
        self.conv_g = nn.Conv2d(1024, 3, kernel_size=3, stride=2)
        self.conv_b = nn.Conv2d(1024, 3, kernel_size=3, stride=2)

    def forward(self, images, features):
        x = self.avgpool(features)

        x = self.conv2(x)
        x = self.relu(x)

        trans_r = self.conv_r(x)
        trans_g = self.conv_g(x)
        trans_b = self.conv_b(x)

        # normalize??
        trans_r = trans_r / sum(trans_r, dim=1, keepdim=True)
        trans_g = trans_g / sum(trans_g, dim=1, keepdim=True)
        trans_b = trans_b / sum(trans_b, dim=1, keepdim=True)

        corr_r = sum(trans_r * images, dim=1, keepdim=True)
        corr_g = sum(trans_g * images, dim=1, keepdim=True)
        corr_b = sum(trans_b * images, dim=1, keepdim=True)

        out = cat([corr_r, corr_g, corr_b], dim=1)

        return out


class GlobalColorUnet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(GlobalColorUnet, self).__init__()

        self.Unet = UNet()
        self.GlobalColorCorrector = GlobalColorCorrector()

    def forward(self, x):
        out_init, features = self.Unet(x)
        out = self.GlobalColorCorrector(out_init, features)
        return out, out_init
