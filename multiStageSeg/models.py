import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class roughSegmenter(nn.Module):
    def __init__(self, filter_factor=2, n_classes=6, in_channels=3, p_drop=0.5):
        super(roughSegmenter, self).__init__()
        ff = filter_factor # filter factor (easy net scaling)
        # Input --> (3, 64, 128, 128)
        # conv layers set 1 - down 1
        self.c1 = nn.Conv3d(in_channels=in_channels, out_channels=int(32*ff), kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(int(32*ff))
        self.drop1 = nn.Dropout3d(p=p_drop)
        self.c2 = nn.Conv3d(in_channels=int(32*ff), out_channels=int(32*ff), kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(int(32*ff))
        self.drop2 = nn.Dropout3d(p=p_drop)
        self.sc1 = nn.Conv3d(in_channels=in_channels, out_channels=int(32*ff), kernel_size=(1,1,1))
        self.sc1_bn = nn.BatchNorm3d(int(32*ff))
        # conv layers set 2 - down 2
        self.c3 = nn.Conv3d(in_channels=int(32*ff), out_channels=int(64*ff), kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(int(64*ff))
        self.drop3 = nn.Dropout3d(p=p_drop)
        self.c4 = nn.Conv3d(in_channels=int(64*ff), out_channels=int(64*ff), kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(int(64*ff))
        self.drop4 = nn.Dropout3d(p=p_drop)
        self.sc2 = nn.Conv3d(in_channels=int(32*ff), out_channels=int(64*ff), kernel_size=(1,1,1))
        self.sc2_bn = nn.BatchNorm3d(int(64*ff))
        # conv layers set 3 - base
        self.c5 = nn.Conv3d(in_channels=int(64*ff), out_channels=int(64*ff), kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(int(64*ff))
        self.drop5 = nn.Dropout3d(p=p_drop)
        self.c6 = nn.Conv3d(in_channels=int(64*ff), out_channels=int(64*ff), kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm3d(int(64*ff))
        self.drop6 = nn.Dropout3d(p=p_drop)
        #self.sc3 = nn.Conv3d(in_channels=int(64*ff), out_channels=int(64*ff), kernel_size=(1,1,1))
        #self.sc3_bn = nn.BatchNorm3d(int(64*ff))
        # transpose convolution
        self.tc1 = nn.ConvTranspose3d(in_channels=int(64*ff), out_channels=int(64*ff), kernel_size=(2,2,2), stride=(2,2,2))
        self.tc1_bn = nn.BatchNorm3d(int(64*ff))
        self.drop_tc1 = nn.Dropout3d(p=p_drop)
        # conv layer set 4 - up 1
        self.c7 = nn.Conv3d(in_channels=int(64*ff), out_channels=int(32*ff), kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm3d(int(32*ff))
        self.drop7 = nn.Dropout3d(p=p_drop)
        self.c8 = nn.Conv3d(in_channels=int(32*ff), out_channels=int(32*ff), kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm3d(int(32*ff))
        self.drop8 = nn.Dropout3d(p=p_drop)
        self.sc4 = nn.Conv3d(in_channels=int(64*ff), out_channels=int(32*ff), kernel_size=(1,1,1))
        self.sc4_bn = nn.BatchNorm3d(int(32*ff))
        # upsample 2
        self.tc2 = nn.ConvTranspose3d(in_channels=int(32*ff), out_channels=int(32*ff), kernel_size=(2,2,2), stride=(2,2,2))
        self.tc2_bn = nn.BatchNorm3d(int(32*ff))
        self.drop_tc2 = nn.Dropout3d(p=p_drop)
        # conv layer set 5 - up 2
        self.c9 = nn.Conv3d(in_channels=int(32*ff), out_channels=int(16*ff), kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm3d(int(16*ff))
        self.drop9 = nn.Dropout3d(p=p_drop)
        self.c10 = nn.Conv3d(in_channels=int(16*ff), out_channels=int(16*ff), kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm3d(int(16*ff))
        self.drop10 = nn.Dropout3d(p=p_drop)
        self.sc5 = nn.Conv3d(in_channels=int(32*ff), out_channels=int(16*ff), kernel_size=(1,1,1))
        self.sc5_bn = nn.BatchNorm3d(int(16*ff))
        # prediction convolution
        self.pred = nn.Conv3d(in_channels=int(16*ff), out_channels=int(n_classes), kernel_size=1)

    @torch.cuda.amp.autocast()
    def forward(self, x):
        # Down block 1
        down1 = F.relu(self.sc1_bn(self.sc1(x)))
        x = F.relu(self.bn1(self.c1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.c2(x)))
        x = self.drop2(x)
        down1 += x
        x = F.max_pool3d(down1, (2,2,2))

        # Down block 2
        down2 = F.relu(self.sc2_bn(self.sc2(x)))
        x = F.relu(self.bn3(self.c3(x)))
        x = self.drop3(x)
        x = F.relu(self.bn4(self.c4(x)))
        x = self.drop4(x)
        down2 += x
        base = F.max_pool3d(down2, (2,2,2))

        # Base block
        x = F.relu(self.bn5(self.c5(base)))
        x = self.drop5(x)
        x = F.relu(self.bn6(self.c6(x)))
        x = self.drop6(x)
        x += base

        # Transpose 1
        x = self.drop_tc1(F.relu(self.tc1_bn(self.tc1(x)))) + F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)

        # Up block 1
        x += down2
        res_skip = F.relu(self.sc4_bn(self.sc4(x)))
        x = F.relu(self.bn7(self.c7(x)))
        x = self.drop7(x)
        x = F.relu(self.bn8(self.c8(x)))
        x = self.drop8(x)
        x += res_skip

        # Upsample 2
        x = self.drop_tc2(F.relu(self.tc2_bn(self.tc2(x)))) + F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        
        # Up block 2
        x += down1
        res_skip = F.relu(self.sc5_bn(self.sc5(x)))
        x = F.relu(self.bn9(self.c9(x)))
        x = self.drop9(x)
        x = F.relu(self.bn10(self.c10(x)))
        x = self.drop10(x)
        x += res_skip

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
