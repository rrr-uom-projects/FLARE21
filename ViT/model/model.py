"""
Vision Transformer Model for Dense Predictions

https://arxiv.org/abs/2103.13413
https://github.com/intel-isl/DPT

"""
import torch
import torch.nn as nn

from .vit import forward_vit
from .blocks import (_make_encoder, 
    FeatureFusionBlock, FeatureFusionBlock_custom, 
    Interpolate)


def _make_fusion_block(features, use_bn):
    #@ Combine patches and project to image for decoder
    return FeatureFusionBlock_custom(features, nn.ReLU(False),
            deconv=False, bn=use_bn,
            expand=False, align_corners=True)


class BaseModel(nn.Module):
    def __init__(self, head, features=256, backbone="vitb_rn50_384", readout='project', 
        channels_last=False, use_bn=False, enable_attention_hooks=False):
        super().__init__()
        self.channels_last = channels_last
        
        #!! Hooks will depend on the backbone we decide to use 
        hooks ={
            "vitb_rn50_384": [0, 1, 8, 11]
        } 

        self.pretrained, self.scratch = _make_encoder(
            backbone, 
            features, 
            use_pretrained=False, 
            groups=1,
            expand=False,
            exportable=False,
            hooks = hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks
        )

        #* Fuse pathc embeddings -> project to image
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head

    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out

class SegmentationModel(BaseModel):
    def __init__(self, num_classes, path=None, **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256
        kwargs["use_bn"] = True

        #! Output head
        head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )
