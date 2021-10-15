import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Softmax
import sys
sys.path.append("..")
from roi_example.model_pytorch_parts import roi_pooling_2d

from .utils import _SimpleSegmentationModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class bingo_cc(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, recurrence = 2, 
        patch_size = 1536, mini_patch_size = 384, output_stride=8, bingo=True):
        super(bingo_cc, self).__init__()
        inter_channels = in_channels // 4
        self.patch_size = patch_size
        self.mini_patch_size = mini_patch_size
        self.output_stride = output_stride
        self.my_batchsize = 16
        self.bingo = bingo


        self.recurrence = recurrence
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),nn.ReLU(inplace=False))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),nn.ReLU(inplace=False))

        self.cat = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)

            )
        
        if(self.bingo):
            self.bingo_process = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                nn.GroupNorm(num_channels=256, num_groups=16),
                nn.ReLU(inplace=False)
            )
            self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.classifier = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # self.grads = {}
        self._init_weight()

    def forward(self, feature):
        mini_patch_size = self.mini_patch_size

        output = self.conva(feature['out'])
        for i in range(self.recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.cat(torch.cat([feature['out'], output], 1)).contiguous()
        if(self.bingo):
            batch = output.shape[0] // self.my_batchsize
            M = int(self.my_batchsize ** 0.5)
            fmap_size = self.mini_patch_size//self.output_stride
            output_big = output.view(batch, self.my_batchsize, 512, fmap_size, fmap_size).contiguous()
            output_big = output_big.permute(0, 1, 3, 4, 2).contiguous().view(batch, M, M, fmap_size, fmap_size, 512).permute(0, 1, 3, 2, 4, 5).contiguous().view(
                batch, M * fmap_size, M * fmap_size, 512).permute(0, 3, 1, 2).contiguous()

            output_big = self.bingo_process(output_big)

            
            output_big = output_big.contiguous()
            batchsize = output_big.shape[0]
            fmap_channel = output_big.shape[1]

            output_big_post = torch.Tensor(self.my_batchsize * batchsize, fmap_channel, fmap_size, fmap_size).to(device)
            for k in range(batchsize):
                for i in range(M):
                    for j in range(M):
                        output_big_post[k * self.my_batchsize + i * M + j, :, :, :] = output_big[k, :, i * fmap_size:(i + 1) * fmap_size, j * fmap_size:(j + 1) * fmap_size]
            output_big_post = output_big_post.contiguous()
        else:
            output_big_post = output

        # w = mini_patch_size//self.output_stride
        # h = mini_patch_size//self.output_stride
        # rois_ = torch.FloatTensor([[0, 0, w - 1, h - 1],
        #                            [w, 0, 2 * w - 1, h - 1],
        #                            [2 * w, 0, 3 * w - 1, h - 1],
        #                            [3 * w, 0, 4 * w - 1, h - 1],
        #                            [0, h, w - 1, 2 * h - 1],
        #                            [w, h, 2 * w - 1, 2 * h - 1],
        #                            [2 * w, h, 3 * w - 1, 2 * h - 1],
        #                            [3 * w, h, 4 * w - 1, 2 * h - 1],
        #                            [0, 2 * h, w - 1, 3 * h - 1],
        #                            [w, 2 * h, 2 * w - 1, 3 * h - 1],
        #                            [2 * w, 2 * h, 3 * w - 1, 3 * h - 1],
        #                            [3 * w, 2 * h, 4 * w - 1, 3 * h - 1],
        #                            [0, 3 * h, w - 1, 4 * h - 1],
        #                            [w, 3 * h, 2 * w - 1, 4 * h - 1],
        #                            [2 * w, 3 * h, 3 * w - 1, 4 * h - 1],
        #                            [3 * w, 3 * h, 4 * w - 1, 4 * h - 1]])
        # rois = torch.FloatTensor([0])
        # for i in range(output_big.shape[0]):
        #     index = torch.FloatTensor([i]).repeat(16, 1)
        #     temp = torch.cat([index, rois_], dim=1)
        #     if (i == 0):
        #         rois = temp
        #     else:
        #         rois = torch.cat([rois, temp], dim=0)
        # rois = rois.cuda()
        # output_big_post = roi_pooling_2d(output_big, rois, (w, h), spatial_scale=1.0).contiguous()


        # output_final = self.classifier(torch.cat([output, output_big_post], 1))
        output_final = self.classifier(output_big_post)



        return output_final



    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)
     # return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)
class CrissCrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.classifier = nn.Sequential(
        #     # nn.Conv2d(304, 256, 3, padding=1, bias=False),
        #     nn.Conv2d(in_dim, 256, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, num_classes, 1)
        # )
        self._init_weight()
    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())

        return self.gamma*(out_H + out_W) + x
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class bingo_deeplabplus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36],
                patch_size = 1536, mini_patch_size = 384, output_stride=4, bingo=True):
        super(bingo_deeplabplus, self).__init__()
        self.patch_size = patch_size
        self.mini_patch_size = mini_patch_size
        self.output_stride = output_stride
        self.my_batchsize = 16
        self.bingo = bingo

        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.bingo_process = nn.Sequential(
            nn.Conv2d(304, 304, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.GroupNorm(num_channels=304, num_groups=16),
            nn.ReLU(inplace=False)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        


        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        output = torch.cat([low_level_feature, output_feature], dim=1).contiguous()

        if(self.bingo):
            fmap_size = self.mini_patch_size//self.output_stride
            batch = output.shape[0] // self.my_batchsize
            M = int(self.my_batchsize ** 0.5)

            output_big = output.view(batch, self.my_batchsize, 304, fmap_size, fmap_size).contiguous()
            output_big = output_big.permute(0, 1, 3, 4, 2).contiguous().view(batch, 
                M, M, fmap_size, fmap_size, 304).permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, 
                M * fmap_size, M * fmap_size, 304).permute(0, 3, 1, 2).contiguous()

            output_big = self.bingo_process(output_big)

            
            output_big = output_big.contiguous()
            batchsize = output_big.shape[0]
            fmap_channel = output_big.shape[1]

            output_big_post = torch.Tensor(self.my_batchsize * batchsize, fmap_channel, fmap_size, fmap_size).to(device)
            for k in range(batchsize):
                for i in range(M):
                    for j in range(M):
                        output_big_post[k * self.my_batchsize + i * M + j, :, :, :] = output_big[k, :, 
                        i * fmap_size:(i + 1) * fmap_size, j * fmap_size:(j + 1) * fmap_size]
            output_big_post = output_big_post.contiguous()
        else:
            output_big_post = output



        return self.classifier(output_big_post)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class bingo_deeplab(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(bingo_deeplab, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier(feature['out'])

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                                module.out_channels,
                                                module.kernel_size,
                                                module.stride,
                                                module.padding,
                                                module.dilation,
                                                module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module



if __name__ == '__main__':
    model = MyModule3(64, 64, 2, recurrence = 2, patch_size = 1536, mini_patch_size = 384, output_stride=8)
    x = torch.randn(2, 64, 48, 48)
    out = model(x)
