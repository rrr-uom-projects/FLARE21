import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class light_segmenter(nn.Module):
    def __init__(self, n_classes=7, in_channels=1, p_drop=0.25):
        super(light_segmenter, self).__init__()
        # Input --> (in_channels, 96, 192, 192)        
        # conv layers set 1 - down 1
        self.c1 = nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.drop1 = nn.Dropout3d(p=p_drop)
        self.c2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.drop2 = nn.Dropout3d(p=p_drop)
        self.sc1 = nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=(1,1,1))
        self.sc1_bn = nn.BatchNorm3d(32)
        # conv layers set 2 - down 2
        self.c3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.drop3 = nn.Dropout3d(p=p_drop)
        self.c4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(64)
        self.drop4 = nn.Dropout3d(p=p_drop)
        self.sc2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(1,1,1))
        self.sc2_bn = nn.BatchNorm3d(64)
        # conv layers set 3 - down 3
        self.c5 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(128)
        self.drop5 = nn.Dropout3d(p=p_drop)
        self.c6 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm3d(128)
        self.drop6 = nn.Dropout3d(p=p_drop)
        self.sc3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1,1,1))
        self.sc3_bn = nn.BatchNorm3d(128)
        # conv layers set 4 - base
        self.c7 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm3d(256)
        self.drop7 = nn.Dropout3d(p=p_drop)
        self.c8 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm3d(256)
        self.drop8 = nn.Dropout3d(p=p_drop)
        self.sc4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1,1,1))
        self.sc4_bn = nn.BatchNorm3d(256)
        # upsample convolution
        self.rc1 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.rc1_bn = nn.BatchNorm3d(256)
        self.rc1_drop = nn.Dropout3d(p=p_drop)
        # conv layer set 5 - up 1
        self.c9 = nn.Conv3d(in_channels=128+256, out_channels=128, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm3d(128)
        self.drop9 = nn.Dropout3d(p=p_drop)
        self.c10 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm3d(128)
        self.drop10 = nn.Dropout3d(p=p_drop)
        self.sc5 = nn.Conv3d(in_channels=128+256, out_channels=128, kernel_size=(1,1,1))
        self.sc5_bn = nn.BatchNorm3d(128)
        # upsample 2
        self.rc2 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.rc2_bn = nn.BatchNorm3d(128)
        self.rc2_drop = nn.Dropout3d(p=p_drop)
        # conv layer set 6 - up 2
        self.c11 = nn.Conv3d(in_channels=64+128, out_channels=64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm3d(64)
        self.drop11 = nn.Dropout3d(p=p_drop)
        self.c12 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm3d(64)
        self.drop12 = nn.Dropout3d(p=p_drop)
        self.sc6 = nn.Conv3d(in_channels=64+128, out_channels=64, kernel_size=(1,1,1))
        self.sc6_bn = nn.BatchNorm3d(64)
        # upsample 3
        self.rc3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.rc3_bn = nn.BatchNorm3d(64)
        self.rc3_drop = nn.Dropout3d(p=p_drop)
        # conv layer set 7 - up 3
        self.c13 = nn.Conv3d(in_channels=32+64, out_channels=32, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm3d(32)
        self.drop13 = nn.Dropout3d(p=p_drop)
        self.c14 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn14 = nn.BatchNorm3d(32)
        self.drop14 = nn.Dropout3d(p=p_drop)
        self.sc7 = nn.Conv3d(in_channels=32+64, out_channels=32, kernel_size=(1,1,1))
        self.sc7_bn = nn.BatchNorm3d(32)
        # prediction convolution
        self.pred = nn.Conv3d(in_channels=32, out_channels=int(n_classes), kernel_size=1)

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
        x = F.max_pool3d(down2, (2,2,2))
        # Down block 3
        down3 = F.relu(self.sc3_bn(self.sc3(x)))
        x = F.relu(self.bn5(self.c5(x)))
        x = self.drop5(x)
        x = F.relu(self.bn6(self.c6(x)))
        x = self.drop6(x)
        down3 += x
        base = F.max_pool3d(down3, (2,2,2))
        # Base block
        x = F.relu(self.bn7(self.c7(base)))
        x = self.drop7(x)
        x = F.relu(self.bn8(self.c8(x)))
        x = self.drop8(x)
        x += F.relu(self.sc4_bn(self.sc4(base)))
        # Upsample 1
        x = self.rc1_drop(F.relu(self.rc1_bn(self.rc1(F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)))))
        # Up bloak 1
        x = torch.cat((x, down3), dim=1)
        res_skip = F.relu(self.sc5_bn(self.sc5(x)))
        x = F.relu(self.bn9(self.c9(x)))
        x = self.drop9(x)
        x = F.relu(self.bn10(self.c10(x)))
        x = self.drop10(x)
        x += res_skip
        # Upsample 2
        x = self.rc2_drop(F.relu(self.rc2_bn(self.rc2(F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)))))
        # Up block 2
        x = torch.cat((x, down2), dim=1)
        res_skip = F.relu(self.sc6_bn(self.sc6(x)))
        x = F.relu(self.bn11(self.c11(x)))
        x = self.drop11(x)
        x = F.relu(self.bn12(self.c12(x)))
        x = self.drop12(x)
        x += res_skip
        # Upsample 2
        x = self.rc3_drop(F.relu(self.rc3_bn(self.rc3(F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)))))
        # Up block 2
        x = torch.cat((x, down1), dim=1)
        res_skip = F.relu(self.sc7_bn(self.sc7(x)))
        x = F.relu(self.bn13(self.c13(x)))
        x = self.drop13(x)
        x = F.relu(self.bn14(self.c14(x)))
        x = self.drop14(x)
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

class yolo_segmenter(nn.Module):
    def __init__(self, n_classes=7, in_channels=1, p_drop=0.25):
        super(yolo_segmenter, self).__init__()
        # Input --> (in_channels, 96, 256, 256)
        self.yolo_input_conv = nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=(7,7,7), padding=(3,3,3), stride=(1,2,2))
        self.yolo_bn = nn.BatchNorm3d(32)
        self.yolo_drop = nn.Dropout3d(p=p_drop)
        # conv layers set 1 - down 1
        self.c1 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.drop1 = nn.Dropout3d(p=p_drop)
        self.c2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.drop2 = nn.Dropout3d(p=p_drop)
        self.sc1 = nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=(1,1,1))
        self.sc1_bn = nn.BatchNorm3d(32)
        # conv layers set 2 - down 2
        self.c3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.drop3 = nn.Dropout3d(p=p_drop)
        self.c4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(64)
        self.drop4 = nn.Dropout3d(p=p_drop)
        self.sc2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(1,1,1))
        self.sc2_bn = nn.BatchNorm3d(64)
        # conv layers set 3 - down 3
        self.c5 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(128)
        self.drop5 = nn.Dropout3d(p=p_drop)
        self.c6 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm3d(128)
        self.drop6 = nn.Dropout3d(p=p_drop)
        self.sc3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1,1,1))
        self.sc3_bn = nn.BatchNorm3d(128)
        # conv layers set 4 - base
        self.c7 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm3d(256)
        self.drop7 = nn.Dropout3d(p=p_drop)
        self.c8 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm3d(256)
        self.drop8 = nn.Dropout3d(p=p_drop)
        self.sc4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1,1,1))
        self.sc4_bn = nn.BatchNorm3d(256)
        # upsample convolution
        self.rc1 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.rc1_bn = nn.BatchNorm3d(256)
        self.rc1_drop = nn.Dropout3d(p=p_drop)
        # conv layer set 5 - up 1
        self.c9 = nn.Conv3d(in_channels=128+256, out_channels=128, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm3d(128)
        self.drop9 = nn.Dropout3d(p=p_drop)
        self.c10 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm3d(128)
        self.drop10 = nn.Dropout3d(p=p_drop)
        self.sc5 = nn.Conv3d(in_channels=128+256, out_channels=128, kernel_size=(1,1,1))
        self.sc5_bn = nn.BatchNorm3d(128)
        # upsample 2
        self.rc2 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.rc2_bn = nn.BatchNorm3d(128)
        self.rc2_drop = nn.Dropout3d(p=p_drop)
        # conv layer set 6 - up 2
        self.c11 = nn.Conv3d(in_channels=64+128, out_channels=64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm3d(64)
        self.drop11 = nn.Dropout3d(p=p_drop)
        self.c12 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm3d(64)
        self.drop12 = nn.Dropout3d(p=p_drop)
        self.sc6 = nn.Conv3d(in_channels=64+128, out_channels=64, kernel_size=(1,1,1))
        self.sc6_bn = nn.BatchNorm3d(64)
        # upsample 3
        self.rc3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.rc3_bn = nn.BatchNorm3d(64)
        self.rc3_drop = nn.Dropout3d(p=p_drop)
        # conv layer set 7 - up 3
        self.c13 = nn.Conv3d(in_channels=32+64, out_channels=32, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm3d(32)
        self.drop13 = nn.Dropout3d(p=p_drop)
        self.c14 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn14 = nn.BatchNorm3d(32)
        self.drop14 = nn.Dropout3d(p=p_drop)
        self.sc7 = nn.Conv3d(in_channels=32+64, out_channels=32, kernel_size=(1,1,1))
        self.sc7_bn = nn.BatchNorm3d(32)
        # upsample 4
        self.rc4 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.rc4_bn = nn.BatchNorm3d(32)
        self.rc4_drop = nn.Dropout3d(p=p_drop)
        # prediction convolution
        self.pred = nn.Conv3d(in_channels=32, out_channels=int(n_classes), kernel_size=1)

    @torch.cuda.amp.autocast()
    def forward(self, x):
        # yolo conv
        x = F.relu(self.yolo_bn(self.yolo_input_conv(x)))
        x = self.yolo_drop(x)
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
        x = F.max_pool3d(down2, (2,2,2))
        # Down block 3
        down3 = F.relu(self.sc3_bn(self.sc3(x)))
        x = F.relu(self.bn5(self.c5(x)))
        x = self.drop5(x)
        x = F.relu(self.bn6(self.c6(x)))
        x = self.drop6(x)
        down3 += x
        base = F.max_pool3d(down3, (2,2,2))
        # Base block
        x = F.relu(self.bn7(self.c7(base)))
        x = self.drop7(x)
        x = F.relu(self.bn8(self.c8(x)))
        x = self.drop8(x)
        x += F.relu(self.sc4_bn(self.sc4(base)))
        # Upsample 1
        x = self.rc1_drop(F.relu(self.rc1_bn(self.rc1(F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)))))
        # Up bloak 1
        x = torch.cat((x, down3), dim=1)
        res_skip = F.relu(self.sc5_bn(self.sc5(x)))
        x = F.relu(self.bn9(self.c9(x)))
        x = self.drop9(x)
        x = F.relu(self.bn10(self.c10(x)))
        x = self.drop10(x)
        x += res_skip
        # Upsample 2
        x = self.rc2_drop(F.relu(self.rc2_bn(self.rc2(F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)))))
        # Up block 2
        x = torch.cat((x, down2), dim=1)
        res_skip = F.relu(self.sc6_bn(self.sc6(x)))
        x = F.relu(self.bn11(self.c11(x)))
        x = self.drop11(x)
        x = F.relu(self.bn12(self.c12(x)))
        x = self.drop12(x)
        x += res_skip
        # Upsample 3
        x = self.rc3_drop(F.relu(self.rc3_bn(self.rc3(F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)))))
        # Up block 3
        x = torch.cat((x, down1), dim=1)
        res_skip = F.relu(self.sc7_bn(self.sc7(x)))
        x = F.relu(self.bn13(self.c13(x)))
        x = self.drop13(x)
        x = F.relu(self.bn14(self.c14(x)))
        x = self.drop14(x)
        x += res_skip
        # Upsample 4
        x = self.rc4_drop(F.relu(self.rc4_bn(self.rc4(F.interpolate(x, scale_factor=(1,1,1,2,2), mode='trilinear', align_corners=False)))))
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
