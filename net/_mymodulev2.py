import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Softmax

from .utils import _SimpleSegmentationModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyModule2(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, recurrence = 2, patch_size = 1536, mini_patch_size = 384, output_stride=8):
        super(MyModule2, self).__init__()
        inter_channels = in_channels // 4
        self.patch_size = patch_size
        self.mini_patch_size = mini_patch_size
        self.output_stride = output_stride

        self.recurrence = recurrence
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),nn.ReLU(inplace=False))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),nn.ReLU(inplace=False))

        self.link = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
            # nn.Dropout2d(0.1),
            # nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=7, dilation=7, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self._init_weight()

    def forward(self, feature):

        mini_patch_size = self.mini_patch_size

        output = self.conva(feature['out'])
        for i in range(self.recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.link(torch.cat([feature['out'], output], 1))

        my_outputs = torch.Tensor(1, 512, self.patch_size//self.output_stride, self.patch_size//self.output_stride).to(device)
        for i in range(4):
            for j in range(4):
                my_outputs[0, :, i * mini_patch_size//self.output_stride:(i + 1) * mini_patch_size//self.output_stride,
                j * mini_patch_size//self.output_stride:(j + 1) * mini_patch_size//self.output_stride] = output[i * 4 + j, :, :, :]
        my_outputs = self.classifier(my_outputs)

        return my_outputs

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



if __name__ == '__main__':
    model = CrissCrossAttention(64)
    x = torch.randn(2, 64, 5, 6)
    out = model(x)
    print(out.shape)