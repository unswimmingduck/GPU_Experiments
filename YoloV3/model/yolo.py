from collections import OrderedDict

import torch
import torch.nn as nn

from backbone import DarkNet53

from blocks import DBL

class Yolo_V3(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained = False):
        super(Yolo_V3, self).__init__()
        #---------------------------------------------------#   
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone = DarkNet53()
        if pretrained:
            self.backbone.load_state_dict(torch.load("model_data/darknet53_backbone_weights.pth"))

        #---------------------------------------------------#
        #   out_filters : [64, 128, 256, 512, 1024]
        #---------------------------------------------------#
        # out_filters = self.backbone.layers_out_filters

        #------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        #------------------------------------------------------------------------#

        self.branch1_DBL5 = DBL(5, 1024, 512, [1, 3, 1, 3, 1], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0])
        self.branch2_DBL5 = DBL(5, 768, 256, [1, 3, 1, 3, 1], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0])
        self.branch3_DBL5 = DBL(5, 384, 128, [1, 3, 1, 3, 1], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0])
        
        self.DBL_upsample_1 = nn.Sequential(DBL(1, 512, 256, [1], [1], [0]),
                                            nn.Upsample(scale_factor=2, mode='nearest'))
        self.DBL_upsample_2 = nn.Sequential(DBL(1, 256, 128, [1], [1], [0]),
                                            nn.Upsample(scale_factor=2, mode='nearest'))

        self.last_layer1 = nn.Sequential(DBL(1, 512, 1024, [3], [1], [1]),
                                         nn.Conv2d(1024, len(anchors_mask[0]) * (num_classes + 5), kernel_size=1, stride=1, padding=0, bias=True))
        self.last_layer2 = nn.Sequential(DBL(1, 256, 512, [3], [1], [1]),
                                         nn.Conv2d(512, len(anchors_mask[1]) * (num_classes + 5), kernel_size=1, stride=1, padding=0, bias=True))
        self.last_layer3 = nn.Sequential(DBL(1, 128, 256, [3], [1], [1]),
                                         nn.Conv2d(256, len(anchors_mask[2]) * (num_classes + 5), kernel_size=1, stride=1, padding=0, bias=True))


    def forward(self, x):
        #---------------------------------------------------#   
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256；26,26,512；13,13,1024
        #---------------------------------------------------#
        x2, x1, x0 = self.backbone(x)

        #---------------------------------------------------#
        #   第一个特征层
        #   out0 = (batch_size,255,13,13)
        #---------------------------------------------------#
        # 13,13,1024 -> 13,13,512
        x0 = self.branch1_DBL5(x0)
        out0 = self.last_layer1(x0)
        # 13,13,512 -> 26,26,256
        x1_merge = self.DBL_upsample_1(x0)

        # 26,26,256 + 26,26,512 = 26,26,768
        merge1 = torch.cat([x1_merge, x1], 1)
        # 26,26,768 -> 26,26,256
        merge1 = self.branch2_DBL5(merge1)
        out1 = self.last_layer2(merge1)
        # 26,26,256 -> 52,52,128
        x2_merge = self.DBL_upsample_2(merge1)
        
        # 52,52,128 + 52,52,256 = 52,52,384
        merge2 = torch.cat([x2_merge, x2], 1)
        # 52,52,384 -> 52,52,128
        merge2 = self.branch3_DBL5(merge2)
        out2 = self.last_layer3(merge2)

        return out0, out1, out2

# model = Yolo_V3([[5,5,5],[5,5,5],[5,5,5]],5)
# model.forward(torch.randn(1,3,416,416))