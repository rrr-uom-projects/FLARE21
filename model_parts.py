import torch.nn as nn
import torch.nn.functional as F

class conv_module(nn.Module):
    def __init__(self, in_channels, out_channels, p_drop=0.25):
        super(conv_module, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
        )
        if in_channels != out_channels:
            self.res_conv = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1,1)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        if self.res_conv is not None:
            return self.double_conv(x) + self.res_conv(x)
        else:
            return self.double_conv(x) + x

class resize_conv(nn.Module):
    def __init__(self, in_channels, out_channels, p_drop, scale_factor=(2,2,2)):
        super(resize_conv, self).__init__()
        self.resize_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
        )
        self.scale_factor = scale_factor
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=False)
        return self.resize_conv(x)

class bottleneck_module(nn.Module):
    def __init__(self, in_channels, out_channels, p_drop=0.25):
        super(bottleneck_module, self).__init__()
        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=(1,1,1)),
            nn.BatchNorm3d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
            nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
            nn.Conv3d(in_channels=in_channels//2, out_channels=out_channels, kernel_size=(1,1,1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
        )
        if in_channels != out_channels:
            self.res_conv = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1,1)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        if self.res_conv is not None:
            return self.bottleneck_conv(x) + self.res_conv(x)
        else:
            return self.bottleneck_conv(x) + x

class asym_conv_module(nn.Module):
    def __init__(self, in_channels, out_channels, p_drop=0.25):
        super(asym_conv_module, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1,3,1), padding=(0,1,0)),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1,1,3), padding=(0,0,1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1,3,1), padding=(0,1,0)),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1,1,3), padding=(0,0,1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
        )
        if in_channels != out_channels:
            self.res_conv = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1,1)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        if self.res_conv is not None:
            return self.double_conv(x) + self.res_conv(x)
        else:
            return self.double_conv(x) + x

class asym_resize_conv(nn.Module):
    def __init__(self, in_channels, out_channels, p_drop, scale_factor=(2,2,2)):
        super(asym_resize_conv, self).__init__()
        self.resize_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1,3,1), padding=(0,1,0)),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1,1,3), padding=(0,0,1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
        )
        self.scale_factor = scale_factor
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=False)
        return self.resize_conv(x)

class asym_bottleneck_module(nn.Module):
    def __init__(self, in_channels, out_channels, p_drop=0.25):
        super(asym_bottleneck_module, self).__init__()
        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=(1,1,1)),
            nn.BatchNorm3d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
            nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(1,3,1), padding=(0,1,0)),
            nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(1,1,3), padding=(0,0,1)),
            nn.BatchNorm3d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
            nn.Conv3d(in_channels=in_channels//2, out_channels=out_channels, kernel_size=(1,1,1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
        )
        if in_channels != out_channels:
            self.res_conv = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1,1)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        if self.res_conv is not None:
            return self.bottleneck_conv(x) + self.res_conv(x)
        else:
            return self.bottleneck_conv(x) + x

class bridge_module(nn.Module):
    def __init__(self, channels, layers, p_drop=0.25):
        super(bridge_module, self).__init__()
        self.conv_bridge = nn.ModuleList([])
        for x in range(layers):
            self.conv_bridge.append(nn.Sequential(
                nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(3,3,3), padding=1),
                nn.BatchNorm3d(channels),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=p_drop, inplace=True),
                nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(3,3,3), padding=1),
                nn.BatchNorm3d(channels),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=p_drop, inplace=True),
                ))
    def forward(self, x):
        for brick in self.conv_bridge:
            x = brick(x) + x
        return x