import torch
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
        else:
            self.res_conv = None
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

class transpose_conv(nn.Module):
    def __init__(self, in_channels, out_channels, p_drop, scale_factor=(2,2,2)):
        super(transpose_conv, self).__init__()
        self.transpose_conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=scale_factor, stride=scale_factor), # stride & kernel (1,2,2) gives (D_in, 2*H_in, 2*W_in)
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
        )
    def forward(self, x):
        return self.transpose_conv(x)

class bottleneck_transpose_conv(nn.Module):
    def __init__(self, in_channels, out_channels, p_drop, scale_factor=(2,2,2), compress_factor=2):
        super(bottleneck_transpose_conv, self).__init__()
        self.bottleneck_transpose_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels//compress_factor, kernel_size=(1,1,1)),
            nn.BatchNorm3d(in_channels//compress_factor),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
            nn.ConvTranspose3d(in_channels=in_channels//compress_factor, out_channels=in_channels//compress_factor, kernel_size=scale_factor, stride=scale_factor), # stride & kernel (1,2,2) gives (D_in, 2*H_in, 2*W_in)
            nn.BatchNorm3d(in_channels//compress_factor),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
            nn.Conv3d(in_channels=in_channels//compress_factor, out_channels=out_channels, kernel_size=(1,1,1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
        )
    def forward(self, x):
        return self.bottleneck_transpose_conv(x)

class bottleneck_module(nn.Module):
    def __init__(self, in_channels, out_channels, p_drop=0.25, compress_factor=2):
        super(bottleneck_module, self).__init__()
        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels//compress_factor, kernel_size=(1,1,1)),
            nn.BatchNorm3d(in_channels//compress_factor),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
            nn.Conv3d(in_channels=in_channels//compress_factor, out_channels=in_channels//compress_factor, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(in_channels//compress_factor),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
            nn.Conv3d(in_channels=in_channels//compress_factor, out_channels=out_channels, kernel_size=(1,1,1)),
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
        else:
            self.res_conv = None
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
    def __init__(self, in_channels, out_channels, p_drop=0.25, compress_factor=2):
        super(asym_bottleneck_module, self).__init__()
        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels//compress_factor, kernel_size=(1,1,1)),
            nn.BatchNorm3d(in_channels//compress_factor),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
            nn.Conv3d(in_channels=in_channels//compress_factor, out_channels=in_channels//compress_factor, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.Conv3d(in_channels=in_channels//compress_factor, out_channels=in_channels//compress_factor, kernel_size=(1,3,1), padding=(0,1,0)),
            nn.Conv3d(in_channels=in_channels//compress_factor, out_channels=in_channels//compress_factor, kernel_size=(1,1,3), padding=(0,0,1)),
            nn.BatchNorm3d(in_channels//compress_factor),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
            nn.Conv3d(in_channels=in_channels//compress_factor, out_channels=out_channels, kernel_size=(1,1,1)),
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
        else:
            self.res_conv = None
    def forward(self, x):
        if self.res_conv is not None:
            return self.bottleneck_conv(x) + self.res_conv(x)
        else:
            return self.bottleneck_conv(x) + x

class asym_bottleneck_transpose_conv(nn.Module):
    def __init__(self, in_channels, out_channels, p_drop, scale_factor=(2,2,2), compress_factor=2):
        super(asym_bottleneck_transpose_conv, self).__init__()
        self.asym_bottleneck_transpose_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels//compress_factor, kernel_size=(1,1,1)),
            nn.BatchNorm3d(in_channels//compress_factor),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
            nn.ConvTranspose3d(in_channels=in_channels//compress_factor, out_channels=in_channels//compress_factor, kernel_size=(scale_factor[0],1,1), stride=(scale_factor[0],1,1)), # stride & kernel (1,2,2) gives (D_in, 2*H_in, 2*W_in)
            nn.ConvTranspose3d(in_channels=in_channels//compress_factor, out_channels=in_channels//compress_factor, kernel_size=(1,scale_factor[1],1), stride=(1,scale_factor[1],1)), # stride & kernel (1,2,2) gives (D_in, 2*H_in, 2*W_in)
            nn.ConvTranspose3d(in_channels=in_channels//compress_factor, out_channels=in_channels//compress_factor, kernel_size=(1,1,scale_factor[2]), stride=(1,1,scale_factor[2])), # stride & kernel (1,2,2) gives (D_in, 2*H_in, 2*W_in)
            nn.BatchNorm3d(in_channels//compress_factor),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
            nn.Conv3d(in_channels=in_channels//compress_factor, out_channels=out_channels, kernel_size=(1,1,1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
        )
    def forward(self, x):
        return self.asym_bottleneck_transpose_conv(x)

class kearney_attention(nn.Module):
    def __init__(self, x1_channels, x2_channels):
        super(kearney_attention, self).__init__()
        self.x1_conv_1 = nn.Conv3d(in_channels=x1_channels, out_channels=x1_channels, kernel_size=(1,1,1), padding=0)
        self.x2_conv_1 = nn.Conv3d(in_channels=x2_channels, out_channels=x1_channels, kernel_size=(1,1,1), padding=0)
        self.chain = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=x1_channels, out_channels=x1_channels, kernel_size=(1,1,1), padding=0),
            nn.BatchNorm3d(x1_channels),
            nn.Sigmoid()
        )
    def forward(self, x1, x2):
        x12 = self.x1_conv_1(x1) + self.x2_conv_1(x2)
        x12 = self.chain(x12)
        return x1 * x12

class depthwise_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(3,3,3), stride=(1,1,1), padding=(1,1,1)):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise_separable_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=kernel, stride=stride, padding=padding, groups=in_channels),
            nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1)),
        )
    def forward(self, x):
        return self.depthwise_separable_conv(x)

class depthwiseSep_bottleneck_module(nn.Module):
    def __init__(self, in_channels, out_channels, p_drop=0.25, compress_factor=2):
        super(depthwiseSep_bottleneck_module, self).__init__()
        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels//compress_factor, kernel_size=(1,1,1)),
            nn.BatchNorm3d(in_channels//compress_factor),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
            depthwise_separable_conv(in_channels//compress_factor, in_channels//compress_factor),
            nn.BatchNorm3d(in_channels//compress_factor),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
            nn.Conv3d(in_channels=in_channels//compress_factor, out_channels=out_channels, kernel_size=(1,1,1)),
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
        else:
            self.res_conv = None
    def forward(self, x):
        if self.res_conv is not None:
            return self.bottleneck_conv(x) + self.res_conv(x)
        else:
            return self.bottleneck_conv(x) + x

class asym_depthwise_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(3,3,3), stride=(1,1,1), padding=(1,1,1)):
        super(asym_depthwise_separable_conv, self).__init__()
        self.asym_depthwise_separable_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(kernel[0],1,1), stride=(stride[0],1,1), padding=(padding[0],0,0), groups=in_channels),
            nn.Conv3d(in_channels, in_channels, kernel_size=(1,kernel[1],1), stride=(1,stride[1],1), padding=(0,padding[1],0), groups=in_channels),
            nn.Conv3d(in_channels, in_channels, kernel_size=(1,1,kernel[2]), stride=(1,1,stride[2]), padding=(0,0,padding[2]), groups=in_channels),
            nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1)),
        )
    def forward(self, x):
        return self.asym_depthwise_separable_conv(x)

class asym_depthwiseSep_bottleneck_module(nn.Module):
    def __init__(self, in_channels, out_channels, p_drop=0.25, compress_factor=2):
        super(asym_depthwiseSep_bottleneck_module, self).__init__()
        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels//compress_factor, kernel_size=(1,1,1)),
            nn.BatchNorm3d(in_channels//compress_factor),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
            asym_depthwise_separable_conv(in_channels//compress_factor, in_channels//compress_factor),
            nn.BatchNorm3d(in_channels//compress_factor),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop, inplace=True),
            nn.Conv3d(in_channels=in_channels//compress_factor, out_channels=out_channels, kernel_size=(1,1,1)),
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
        else:
            self.res_conv = None
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

class learnable_WL(nn.Module):
    def __init__(self, num_channels, initial_levels, initial_windows):
        # initial levels and windows are lists of with len==num_channels. These are guesses of the W/L you wish to use -> could be made optional
        super(learnable_WL, self).__init__()
        self.num_channels = num_channels
        # Checks
        try:
            assert(len(initial_levels)==self.num_channels and len(initial_windows)==self.num_channels)
        except AssertionError:
            print("Wrong number of initialised windows/levels...")
            exit(1)
        # Parameter initialisation
        self.levels = nn.Parameter(torch.Tensor(initial_levels) / 3024)
        self.windows = nn.Parameter(torch.Tensor(initial_windows) / 3024)
    
    '''
    def sigmoid(self, x, ch_idx):
        # General sigmoid is of form S(x|a,b) = 1/(1+exp(a(b-x)))
        # use a = 2*ln(9)/w & b = level
        return 1/(1+torch.exp(((2*torch.log(9))/ self.windows[ch_idx])*(self.levels[ch_idx]-x)))
    '''
    def sigmoid(self, x):
        # General sigmoid is of form S(x|a,b) = 1/(1+exp(a(b-x)))
        # use a = 2*ln(9)/w & b = level
        # promote the dimension of windows and levels to use broadcasting
        return 1 / (1 + torch.exp((2 * torch.log(torch.tensor(9)) / self.windows[:, None, None, None]) * (self.levels[:, None, None, None] - x)))

    def forward(self, image):
        # first put the image onto the range [0,1] <-- currently doing proper HU, no WM shift
        image = torch.clamp(image, -1024, 2000) / 3024
        # repeat image along channels dimension
        image = torch.repeat_interleave(image, repeats=self.num_channels, dim=1)
        # apply the sigmoid
        return self.sigmoid(image)
        '''
        # now use the sigmoid for each channel (non-vectorised version)
        ims = []
        for ch_idx in range(self.num_channels):
            im = self.sigmoid(image, ch_idx)
            ims.append(im)
        image = torch.cat(ims, dim=1)
        '''