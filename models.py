import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_parts import conv_module, resize_conv, bottleneck_module, asym_conv_module, asym_resize_conv
from roughSeg.utils import get_number_of_learnable_parameters

# Basic model: Unet-like with residula connections and increased input size: (96, 192, 192)
# Parameters: 8,272,487
# Achieves > 0.95 DSC on all but pancreas (~0.8)
class light_segmenter(nn.Module):
    def __init__(self, n_classes=7, in_channels=1, p_drop=0.25):
        super(light_segmenter, self).__init__()
        # Input --> (in_channels, 96, 192, 192)        
        # conv layers set 1 - down 1
        self.down_conv_1 = conv_module(in_channels=in_channels, out_channels=32, p_drop=p_drop)
        # conv layers set 2 - down 2
        self.down_conv_2 = conv_module(in_channels=32, out_channels=64, p_drop=p_drop)
        # conv layers set 3 - down 3
        self.down_conv_3 = conv_module(in_channels=64, out_channels=128, p_drop=p_drop)
        # conv layers set 4 - base
        self.base_conv = conv_module(in_channels=128, out_channels=256, p_drop=p_drop)
        # upsample convolution and up set 1
        self.upsample_1 = resize_conv(in_channels=256, out_channels=256, p_drop=p_drop)
        self.up_conv_1 = conv_module(in_channels=128+256, out_channels=128, p_drop=p_drop)
        # upsample 2 and up 2
        self.upsample_2 = resize_conv(in_channels=128, out_channels=128, p_drop=p_drop)
        self.up_conv_2 = conv_module(in_channels=64+128, out_channels=64, p_drop=p_drop)
        # upsample and up 3
        self.upsample_3 = resize_conv(in_channels=64, out_channels=64, p_drop=p_drop)
        self.up_conv_3 = conv_module(in_channels=32+64, out_channels=32, p_drop=p_drop)
        # prediction convolution
        self.pred = nn.Conv3d(in_channels=32, out_channels=int(n_classes), kernel_size=1)

    @torch.cuda.amp.autocast()
    def forward(self, x):
        # Down block 1
        down1 = self.down_conv_1(x) 
        x = F.max_pool3d(down1, (2,2,2))
        # Down block 2
        down2 = self.down_conv_2(x)
        x = F.max_pool3d(down2, (2,2,2))
        # Down block 3
        down3 = self.down_conv_3(x)
        x = F.max_pool3d(down3, (2,2,2))
        # Base block
        x = self.base_conv(x)
        # Upsample and up block 1
        x = self.upsample_1(x)
        x = torch.cat((x, down3), dim=1)
        x = self.up_conv_1(x)
        # Upsample 2 and up block 2
        x = self.upsample_2(x)
        x = torch.cat((x, down2), dim=1)
        x = self.up_conv_2(x)
        # Upsample 3
        x = self.upsample_3(x)
        x = torch.cat((x, down1), dim=1)
        x = self.up_conv_3(x)
        # Predict
        return self.pred(x)

    def load_best(self, checkpoint_dir, logger):
        # load previous best weights
        model_dict = self.state_dict()
        state = torch.load(os.path.join(checkpoint_dir, 'best_checkpoint.pytorch'))
        best_checkpoint_dict = state['model_state_dict']
        # remove the 'module.' wrapper
        renamed_dict = OrderedDict()
        for key, value in best_checkpoint_dict.items():
            new_key = key.replace('module.','')
            renamed_dict[new_key] = value
        # identify which layers to grab
        renamed_dict = {k: v for k, v in list(renamed_dict.items()) if k in model_dict}
        model_dict.update(renamed_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("Loaded layers from previous best checkpoint:")
            logger.info([k for k, v in list(renamed_dict.items())])

## Unet-esque model with YOLO inspired input architecture allowing higher resolution input (96,256,256)
# Parameters: 8,319,207
# visually looks good and doesn't take years to train, but performance not yet known
class yolo_segmenter(nn.Module):
    def __init__(self, n_classes=7, in_channels=1, p_drop=0.25):
        super(yolo_segmenter, self).__init__()
        # Input --> (in_channels, 96, 256, 256)
        self.yolo_input_conv = nn.Conv3d(in_channels=in_channels, out_channels=16, kernel_size=(7,7,7), padding=(3,3,3), stride=(1,2,2))
        self.yolo_bn = nn.BatchNorm3d(16)
        self.yolo_drop = nn.Dropout3d(p=p_drop)
        # conv layers set 1 - down 1
        self.down_conv_1 = conv_module(in_channels=16, out_channels=32, p_drop=p_drop)
        # conv layers set 2 - down 2
        self.down_conv_2 = conv_module(in_channels=32, out_channels=64, p_drop=p_drop)
        # conv layers set 3 - down 3
        self.down_conv_3 = conv_module(in_channels=64, out_channels=128, p_drop=p_drop)
        # conv layers set 4 - base
        self.base_conv = conv_module(in_channels=128, out_channels=256, p_drop=p_drop)
        # upsample convolution and up set 1
        self.upsample_1 = resize_conv(in_channels=256, out_channels=256, p_drop=p_drop)
        self.up_conv_1 = conv_module(in_channels=128+256, out_channels=128, p_drop=p_drop)
        # upsample 2 and up 2
        self.upsample_2 = resize_conv(in_channels=128, out_channels=128, p_drop=p_drop)
        self.up_conv_2 = conv_module(in_channels=64+128, out_channels=64, p_drop=p_drop)
        # upsample and up 3
        self.upsample_3 = resize_conv(in_channels=64, out_channels=64, p_drop=p_drop)
        self.up_conv_3 = conv_module(in_channels=32+64, out_channels=32, p_drop=p_drop)
        # upsample 4 and prediction convolution
        self.upsample_4 = resize_conv(in_channels=32, out_channels=32, p_drop=p_drop, scale_factor=(1,2,2))
        self.pred = nn.Conv3d(in_channels=32, out_channels=int(n_classes), kernel_size=1)

    @torch.cuda.amp.autocast()
    def forward(self, x):
        # yolo conv
        x = F.relu(self.yolo_bn(self.yolo_input_conv(x)))
        x = self.yolo_drop(x)
        # Down block 1
        down1 = self.down_conv_1(x) 
        x = F.max_pool3d(down1, (2,2,2))
        # Down block 2
        down2 = self.down_conv_2(x)
        x = F.max_pool3d(down2, (2,2,2))
        # Down block 3
        down3 = self.down_conv_3(x)
        x = F.max_pool3d(down3, (2,2,2))
        # Base block
        x = self.base_conv(x)
        # Upsample and up block 1
        x = self.upsample_1(x)
        x = torch.cat((x, down3), dim=1)
        x = self.up_conv_1(x)
        # Upsample 2 and up block 2
        x = self.upsample_2(x)
        x = torch.cat((x, down2), dim=1)
        x = self.up_conv_2(x)
        # Upsample 3
        x = self.upsample_3(x)
        x = torch.cat((x, down1), dim=1)
        x = self.up_conv_3(x)
        # Upsample 4 and predict
        x = self.upsample_4(x)
        return self.pred(x)

    def load_best(self, checkpoint_dir, logger):
        # load previous best weights
        model_dict = self.state_dict()
        state = torch.load(os.path.join(checkpoint_dir, 'best_checkpoint.pytorch'))
        best_checkpoint_dict = state['model_state_dict']
        # remove the 'module.' wrapper
        renamed_dict = OrderedDict()
        for key, value in best_checkpoint_dict.items():
            new_key = key.replace('module.','')
            renamed_dict[new_key] = value
        # identify which layers to grab
        renamed_dict = {k: v for k, v in list(renamed_dict.items()) if k in model_dict}
        model_dict.update(renamed_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("Loaded layers from previous best checkpoint:")
            logger.info([k for k, v in list(renamed_dict.items())])

## As above with bottleneck modules instead of double convolutions (see ResNet)
# lowers parameters from 8,319,207 -> 5,491607
# again visually looks good, takes same time to train and similar loss curves. -> True test will be at inference time.
class bottleneck_yolo_segmenter(nn.Module):
    def __init__(self, n_classes=7, in_channels=1, p_drop=0.25):
        super(bottleneck_yolo_segmenter, self).__init__()
        # Input --> (in_channels, 96, 256, 256)
        self.yolo_input_conv = nn.Conv3d(in_channels=in_channels, out_channels=16, kernel_size=(7,7,7), padding=(3,3,3), stride=(1,2,2))
        self.yolo_bn = nn.BatchNorm3d(16)
        self.yolo_drop = nn.Dropout3d(p=p_drop)
        # conv layers set 1 - down 1
        self.down_conv_1 = bottleneck_module(in_channels=16, out_channels=32, p_drop=p_drop)
        # conv layers set 2 - down 2
        self.down_conv_2 = bottleneck_module(in_channels=32, out_channels=64, p_drop=p_drop)
        # conv layers set 3 - down 3
        self.down_conv_3 = bottleneck_module(in_channels=64, out_channels=128, p_drop=p_drop)
        # conv layers set 4 - base
        self.base_conv = bottleneck_module(in_channels=128, out_channels=256, p_drop=p_drop)
        # upsample convolution and up set 1
        self.upsample_1 = resize_conv(in_channels=256, out_channels=256, p_drop=p_drop)
        self.up_conv_1 = bottleneck_module(in_channels=128+256, out_channels=128, p_drop=p_drop)
        # upsample 2 and up 2
        self.upsample_2 = resize_conv(in_channels=128, out_channels=128, p_drop=p_drop)
        self.up_conv_2 = bottleneck_module(in_channels=64+128, out_channels=64, p_drop=p_drop)
        # upsample and up 3
        self.upsample_3 = resize_conv(in_channels=64, out_channels=64, p_drop=p_drop)
        self.up_conv_3 = bottleneck_module(in_channels=32+64, out_channels=32, p_drop=p_drop)
        # upsample 4 and prediction convolution
        self.upsample_4 = resize_conv(in_channels=32, out_channels=32, p_drop=p_drop, scale_factor=(1,2,2))
        self.pred = nn.Conv3d(in_channels=32, out_channels=int(n_classes), kernel_size=1)

    @torch.cuda.amp.autocast()
    def forward(self, x):
        # yolo conv
        x = F.relu(self.yolo_bn(self.yolo_input_conv(x)))
        x = self.yolo_drop(x)
        # Down block 1
        down1 = self.down_conv_1(x) 
        x = F.max_pool3d(down1, (2,2,2))
        # Down block 2
        down2 = self.down_conv_2(x)
        x = F.max_pool3d(down2, (2,2,2))
        # Down block 3
        down3 = self.down_conv_3(x)
        x = F.max_pool3d(down3, (2,2,2))
        # Base block
        x = self.base_conv(x)
        # Upsample and up block 1
        x = self.upsample_1(x)
        x = torch.cat((x, down3), dim=1)
        x = self.up_conv_1(x)
        # Upsample 2 and up block 2
        x = self.upsample_2(x)
        x = torch.cat((x, down2), dim=1)
        x = self.up_conv_2(x)
        # Upsample 3
        x = self.upsample_3(x)
        x = torch.cat((x, down1), dim=1)
        x = self.up_conv_3(x)
        # Upsample 4 and predict
        x = self.upsample_4(x)
        return self.pred(x)

    def load_best(self, checkpoint_dir, logger):
        # load previous best weights
        model_dict = self.state_dict()
        state = torch.load(os.path.join(checkpoint_dir, 'best_checkpoint.pytorch'))
        best_checkpoint_dict = state['model_state_dict']
        # remove the 'module.' wrapper
        renamed_dict = OrderedDict()
        for key, value in best_checkpoint_dict.items():
            new_key = key.replace('module.','')
            renamed_dict[new_key] = value
        # identify which layers to grab
        renamed_dict = {k: v for k, v in list(renamed_dict.items()) if k in model_dict}
        model_dict.update(renamed_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("Loaded layers from previous best checkpoint:")
            logger.info([k for k, v in list(renamed_dict.items())])

## As yolo segmenter but with experimental assymetric convolution stacks
# lowers parameters from 8,319,207 -> 2,860,967
# highly doubtful this will work -> could be compressed even smaller with bottlenecking
class asymmetric_yolo_segmenter(nn.Module):
    def __init__(self, n_classes=7, in_channels=1, p_drop=0.25):
        super(asymmetric_yolo_segmenter, self).__init__()
        # Input --> (in_channels, 96, 256, 256)
        self.yolo_input_conv = nn.Conv3d(in_channels=in_channels, out_channels=16, kernel_size=(7,7,7), padding=(3,3,3), stride=(1,2,2))
        self.yolo_bn = nn.BatchNorm3d(16)
        self.yolo_drop = nn.Dropout3d(p=p_drop)
        # conv layers set 1 - down 1
        self.down_conv_1 = asym_conv_module(in_channels=16, out_channels=32, p_drop=p_drop)
        # conv layers set 2 - down 2
        self.down_conv_2 = asym_conv_module(in_channels=32, out_channels=64, p_drop=p_drop)
        # conv layers set 3 - down 3
        self.down_conv_3 = asym_conv_module(in_channels=64, out_channels=128, p_drop=p_drop)
        # conv layers set 4 - base
        self.base_conv = asym_conv_module(in_channels=128, out_channels=256, p_drop=p_drop)
        # upsample convolution and up set 1
        self.upsample_1 = asym_resize_conv(in_channels=256, out_channels=256, p_drop=p_drop)
        self.up_conv_1 = asym_conv_module(in_channels=128+256, out_channels=128, p_drop=p_drop)
        # upsample 2 and up 2
        self.upsample_2 = asym_resize_conv(in_channels=128, out_channels=128, p_drop=p_drop)
        self.up_conv_2 = asym_conv_module(in_channels=64+128, out_channels=64, p_drop=p_drop)
        # upsample and up 3
        self.upsample_3 = asym_resize_conv(in_channels=64, out_channels=64, p_drop=p_drop)
        self.up_conv_3 = asym_conv_module(in_channels=32+64, out_channels=32, p_drop=p_drop)
        # upsample 4 and prediction convolution
        self.upsample_4 = asym_resize_conv(in_channels=32, out_channels=32, p_drop=p_drop, scale_factor=(1,2,2))
        self.pred = nn.Conv3d(in_channels=32, out_channels=int(n_classes), kernel_size=1)

    @torch.cuda.amp.autocast()
    def forward(self, x):
        # yolo conv
        x = F.relu(self.yolo_bn(self.yolo_input_conv(x)))
        x = self.yolo_drop(x)
        # Down block 1
        down1 = self.down_conv_1(x) 
        x = F.max_pool3d(down1, (2,2,2))
        # Down block 2
        down2 = self.down_conv_2(x)
        x = F.max_pool3d(down2, (2,2,2))
        # Down block 3
        down3 = self.down_conv_3(x)
        x = F.max_pool3d(down3, (2,2,2))
        # Base block
        x = self.base_conv(x)
        # Upsample and up block 1
        x = self.upsample_1(x)
        x = torch.cat((x, down3), dim=1)
        x = self.up_conv_1(x)
        # Upsample 2 and up block 2
        x = self.upsample_2(x)
        x = torch.cat((x, down2), dim=1)
        x = self.up_conv_2(x)
        # Upsample 3
        x = self.upsample_3(x)
        x = torch.cat((x, down1), dim=1)
        x = self.up_conv_3(x)
        # Upsample 4 and predict
        x = self.upsample_4(x)
        return self.pred(x)

    def load_best(self, checkpoint_dir, logger):
        # load previous best weights
        model_dict = self.state_dict()
        state = torch.load(os.path.join(checkpoint_dir, 'best_checkpoint.pytorch'))
        best_checkpoint_dict = state['model_state_dict']
        # remove the 'module.' wrapper
        renamed_dict = OrderedDict()
        for key, value in best_checkpoint_dict.items():
            new_key = key.replace('module.','')
            renamed_dict[new_key] = value
        # identify which layers to grab
        renamed_dict = {k: v for k, v in list(renamed_dict.items()) if k in model_dict}
        model_dict.update(renamed_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("Loaded layers from previous best checkpoint:")
            logger.info([k for k, v in list(renamed_dict.items())])

print(get_number_of_learnable_parameters(asymmetric_yolo_segmenter()))