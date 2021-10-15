import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("./roi_example")
from roi_pooling.functions.roi_pooling import ROIPooling2d
# import torch.autograd.Function

def roi_pooling_2d(input, rois, output_size, spatial_scale=1.0):
    return ROIPooling2d.apply(input, rois, output_size, spatial_scale)

class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def roi_pooling_2d(self, input, rois, output_size, spatial_scale=1.0):
        return ROIPooling2d.apply(input, rois, output_size, spatial_scale)
    def forward(self, fea_de, fea_en):
        batch = fea_de.shape[0]


        x1 = (fea_en.shape[3] - fea_de.shape[3]) // 2
        y1 = (fea_en.shape[2] - fea_de.shape[2]) // 2
        x2 = x1 + fea_de.shape[3]
        y2 = y1 + fea_de.shape[2]

        rois_ = torch.FloatTensor([x1, y1, x2, y2]).repeat(batch, 1)
        bbox_index = torch.arange(0, batch).float()
        bbox_index = bbox_index.reshape(batch, 1)
        rois = torch.cat([bbox_index, rois_], dim=1)
        rois = rois.cuda()

        fea_en_cropped = roi_pooling_2d(fea_en, rois, fea_de.shape[2:], 1)
        return torch.cat([fea_en_cropped, fea_de], dim=1)



# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""
#
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.double_conv(x)

class Double_Conv(nn.Module):
    def __init__(self, in_channels, out_channels = 16, n_convs = 2):
        super(Double_Conv, self).__init__()
        self.double_cov = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        return self.double_cov(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        x = self.up(x)
        out = self.conv(x)
        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)