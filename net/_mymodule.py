import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Softmax

# from .utils import _SimpleSegmentationModel


class MyModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, recurrence = 2):
        super(MyModule, self).__init__()
        CC_channels = in_channels // 4
        my_channels = in_channels // 4
        self.recurrence = recurrence
        self.conva = nn.Sequential(nn.Conv2d(in_channels, CC_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(CC_channels),nn.ReLU(inplace=False))
        self.cca = CCAttention(CC_channels)
        self.convb = nn.Sequential(nn.Conv2d(CC_channels, CC_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(CC_channels),nn.ReLU(inplace=False))

        self.my_conv1 = nn.Sequential(nn.Conv2d(in_channels, my_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(my_channels),nn.ReLU(inplace=False))
        self.mya = MyAttention(my_channels)
        self.my_conv2 = nn.Sequential(nn.Conv2d(my_channels, my_channels, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(my_channels), nn.ReLU(inplace=False))

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels+CC_channels+my_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            # nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        self._init_weight()

    def forward(self, feature):

        cc_output = self.conva(feature['out'])
        my_output = self.my_conv1(feature['out'])
        # output = self.conva(feature)#debug

        # print(output.shape)

        for i in range(self.recurrence):
            cc_output = self.cca(cc_output)
        for i in range(self.recurrence):
            my_output = self.mya(my_output)

        # print(output.shape)

        cc_output = self.convb(cc_output)
        my_output = self.my_conv2(my_output)

        # output = self.classifier(torch.cat([feature['out'], cc_output, my_output], 1))
        output = self.classifier(torch.cat([feature['out'], cc_output, my_output], 1))
        # output = self.classifier(torch.cat([feature, output], 1))#debug
        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)
     # temp = -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)
     # print(temp.shape)
     # print(temp)
     # return temp
     # return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)#debug
class CCAttention(nn.Module):
    def __init__(self, in_dim):
        super(CCAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
        self._init_weight()
    def forward(self, x):
        m_batchsize, _, height, width = x.size() #(16, 512, H, W)
        proj_query = self.query_conv(x) #(16, 64, H, W)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        #(16*W, H, 64)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x) #(16, 64, H, W)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        #(16*W, 64, H)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x) #(16, 512, H, W)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        #(16*W, 512, H)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        #(16, H, W, H)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        #(16, H, W, W)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))#(16, H, W, H+W)

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #(16*W, H, H)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        #(16*H, W, W)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        #(16, 512, H, W)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #(16, 512, H, W)
        # print(out_H.size(),out_W.size())

        return self.gamma*(out_H + out_W) + x
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class MyAttention(nn.Module):
    def __init__(self, in_dim):
        super(MyAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

        self._init_weight()
    def forward(self, x):

        m_batchsize, channel, height, width = x.size() #(16, 512, H, W)
        batch_W = m_batchsize // 4
        batch_H = m_batchsize // 4


        proj_query = torch.unsqueeze(self.query_conv(x), 0) #(1, 16, 64, H, W)
        proj_query = proj_query.contiguous().view(1, m_batchsize, -1) #(1, 16, 64*H*W)
        proj_query = proj_query.permute(0,2,1).contiguous().view(1,-1,batch_H,batch_W)
        #(1, 64*H*W, batch_H, batch_W)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(1*batch_W,-1,batch_H).permute(0, 2, 1)
        #(1*batch_W, batch_H, 64*H*W)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(1*batch_H,-1,batch_W).permute(0, 2, 1)
        #(1*batch_H, batch_W, 64*H*W)


        proj_key = torch.unsqueeze(self.key_conv(x), 0) #(1, 16, 64, H, W)
        proj_key = proj_key.contiguous().view(1, m_batchsize, -1)#(1, 16, 64*H*W)
        proj_key = proj_key.permute(0,2,1).contiguous().view(1,-1,batch_H,batch_W)
        #(1, 64*H*W, batch_H, batch_W)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(1*batch_W,-1,batch_H)
        #(1*batch_W, 64*H*W, batch_H)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(1*batch_H,-1,batch_W)
        #(1*batch_H, 64*H*W, batch_H)


        proj_value = torch.unsqueeze(self.value_conv(x), 0) #(1, 16, 512, H, W)
        proj_value = proj_value.contiguous().view(1, m_batchsize, -1)  # (1, 16, 512*H*W)
        proj_value = proj_value.permute(0, 2, 1).contiguous().view(1, -1, batch_H, batch_W)
        #(1, 512*H*W, batch_H, batch_W)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(1*batch_W,-1,batch_H)
        #(1*batch_W, 512*H*W, batch_H)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(1*batch_H,-1,batch_W)
        #(1*batch_H, 512*H*W, batch_W)


        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(1, batch_H, batch_W)).view(1,batch_W,batch_H,batch_H).permute(0,2,1,3)
        #(1, batch_H, batch_W, batch_H)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(1,batch_H,batch_W,batch_W)
        #(1, batch_H, batch_W, batch_W)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        #(1, batch_H, batch_W, batch_H+batch_W)

        att_H = concate[:,:,:,0:batch_H].permute(0,2,1,3).contiguous().view(1*batch_W,batch_H,batch_H)
        #(1*batch_W, batch_H, batch_H)
        att_W = concate[:,:,:,batch_H:batch_H+batch_W].contiguous().view(1*batch_H,batch_W,batch_W)
        #(1*batch_H, batch_W, batch_W)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(1,batch_W,-1,batch_H).permute(0,2,3,1)
        #(1, 512*H*W, batch_H, batch_W)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(1,batch_H,-1,batch_W).permute(0,2,1,3)
        #(1, 512*H*W, batch_H, batch_W)

        out_H = out_H.contiguous().view(1, channel*height*width, m_batchsize).permute(0, 2, 1)#(1, batch, 512*H*W)
        out_H = torch.squeeze(out_H.contiguous().view(1, m_batchsize, channel, height, width))
        out_W = out_W.contiguous().view(1, channel*height*width, m_batchsize).permute(0, 2, 1)
        out_W = torch.squeeze(out_W.contiguous().view(1, m_batchsize, channel, height, width))
        # print(out_H.size(),out_W.size())

        return self.gamma*(out_H + out_W) + x
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



if __name__ == '__main__':
    model = MyAttention(512)
    # model = MyModule(2048, 512, 2)
    x = torch.randn(16, 512, 48, 48)
    out = model(x)
    print(out.shape)