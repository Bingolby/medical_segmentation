import torch
import torch.nn as nn
import torch.nn.functional as F

# from net.aspp import build_aspp
# from net.decoder import build_decoder
from net.backbone import build_backbone

def conv2d(in_channel, out_channel, kernel_size):
    layers = [
        nn.Conv2d(
            in_channel, out_channel, kernel_size, padding=kernel_size // 2, bias=False
        ),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
    ]

    return nn.Sequential(*layers)

def conv1d(in_channel, out_channel):
    layers = [
        nn.Conv1d(in_channel, out_channel, 1, bias=False),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
    ]

    return nn.Sequential(*layers)

class object_context(nn.Module):
    def __init__(self, n_class, feat_channels=[512, 2048]):
        super(object_context, self).__init__()

        # self.backbone = backbone

        ch16, ch32 = feat_channels

        self.L = nn.Conv2d(ch16, n_class, 1)
        self.X = conv2d(ch32, 512, 3)

        self.phi = conv1d(512, 256)
        self.psi = conv1d(512, 256)
        self.delta = conv1d(512, 256)
        self.rho = conv1d(256, 512)
        self.g = conv2d(512 + 512, 512, 1)

        self.out = nn.Conv2d(512, n_class, 1)

        self._init_weight()

        # self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, input, x, low_level_feat, mid_level_feat):
        input_size = input.shape[2:]                                #(520,520)
        # stg16, stg32 = self.backbone(input)[-2:]                  #([1, 768, 65, 65])   ([1, 1024, 65, 65])
        X = self.X(x)                                               #([1, 512, 65, 65])
        L = self.L(mid_level_feat)                                  #([1,   2, 65, 65])
        batch, n_class, height, width = L.shape
        l_flat = L.view(batch, n_class, -1)                         #([1, 2, 4225])
        # M: NKL
        M = torch.softmax(l_flat, -1)                               #([1, 2, 4225])
        channel = X.shape[1]
        X_flat = X.view(batch, channel, -1)                         #([1, 512, 4225])
        # f_k: NCK
        f_k = (M @ X_flat.transpose(1, 2)).transpose(1, 2)          #([1, 512, 2])

        # query: NKD
        query = self.phi(f_k).transpose(1, 2)                       #([1, 2, 256])
        # key: NDL
        key = self.psi(X_flat)                                      #([1, 256, 4225])
        logit = query @ key                                         #([1, 2, 4225])
        # attn: NKL
        attn = torch.softmax(logit, 1)                              #([1, 2, 4225])

        # delta: NDK
        delta = self.delta(f_k)                                     #([1, 256, 2])
        # attn_sum: NDL
        attn_sum = delta @ attn                                     #([1, 256, 4225])
        # x_obj = NCHW
        X_obj = self.rho(attn_sum).view(batch, -1, height, width)   #([1, 512, 65, 65])

        concat = torch.cat([X, X_obj], 1)                           #([1, 1024, 65, 65])
        X_bar = self.g(concat)                                      #([1, 512, 65, 65])
        out = self.out(X_bar)                                       #([1, 2, 65, 65])
        main_out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
                                                                    #([1, 2, 520, 520])
        aux_out = F.interpolate(
            L, size=input_size, mode='bilinear', align_corners=False
        )
        return aux_out, main_out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class OCR(nn.Module):
    def __init__(self, backbone = 'resnet', output_stride = 8, num_classes = 2):
        super(OCR, self).__init__()
        BachNorm = nn.BatchNorm2d
        self.backbone = build_backbone(backbone, output_stride)
        feat_channels = [512, 2048]
        self.object_context = object_context(num_classes, feat_channels)
        # self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        # self.aspp = build_aspp(backbone, output_stride, BachNorm)
        # self.decoder = build_decoder(num_classes, backbone, BachNorm)

    def forward(self, input):
        #print('input: ', input.shape)
        # input_size = input.shape[2:]
        x, low_level_feat, mid_level_feat = self.backbone(input)                            #([16, 2048, 24, 24]) ([16, 256, 96, 96]) ([16, 512, 48, 48])
        aux_out, main_out = self.object_context(input, x, low_level_feat, mid_level_feat)
        # X = self.X(x)
        # L = self.L(mid_level_feat)
        # batch, n_class, height, width = L.shape
        # l_flat = L.view(batch, n_class, -1)
        # M = torch.softmax(l_flat, -1)
        # channel = X.shape[1]
        # X_flat = X.view(batch, channel, -1)
        # f_k = (M @ X_flat.transpose(1, 2)).transpose(1, 2)
        # query = self.phi(f_k).transpose(1, 2)
        # key = self.psi(X_flat)
        # logit = query @ key
        # attn = torch.softmax(logit, 1)
        # delta = self.delta(f_k)
        # attn_sum = delta @ attn
        # X_obj = self.rho(attn_sum).view(batch, -1, height, width)
        # concat = torch.cat([X, X_obj], 1)
        # X_bar = self.g(concat)
        # out = self.out(X_bar)
        # out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        # aux_out = F.interpolate(
        #     L, size=input_size, mode='bilinear', align_corners=False
        # )
        return aux_out, main_out

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.object_context]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield  p



























































