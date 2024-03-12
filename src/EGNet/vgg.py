import torch.nn as nn
import math
# import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import torch.nn.functional as F

# vgg16
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    stage = 1
    for v in cfg:
        if v == 'M':
            stage += 1
            if stage == 6:
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
            else:
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        else:
            if stage == 6:
                # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=4, dilation=4, bias=False)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers

class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()
        self.cfg = {'tun': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 'tun_ex': [512, 512, 512]}
        self.extract = [8, 15, 22, 29] # [3, 8, 15, 22, 29]
        self.extract_ex = [5]
        self.base = nn.ModuleList(vgg(self.cfg['tun'], 3))
        self.base_ex = vgg_ex(self.cfg['tun_ex'], 512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_pretrained_model(self, model):
        self.base.load_state_dict(model)

    def forward(self, x, multi=0):
        tmp_x = []
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.extract:
                tmp_x.append(x)
        x = self.base_ex(x)
        tmp_x.append(x)
        if multi == 1:
            tmp_y = []
            tmp_y.append(tmp_x[0])
            return tmp_y
        else:
            return tmp_x

class vgg_ex(nn.Module):
    def __init__(self, cfg, incs=512, padding=1, dilation=1):
        super(vgg_ex, self).__init__()
        self.cfg = cfg
        layers = []
        for v in self.cfg:
            # conv2d = nn.Conv2d(incs, v, kernel_size=3, padding=4, dilation=4, bias=False)
            conv2d = nn.Conv2d(incs, v, kernel_size=3, padding=padding, dilation=dilation, bias=False)
            layers += [conv2d, nn.ReLU(inplace=True)]
            incs = v
        self.ex = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.ex(x)
        return x

# class vgg16_locate(nn.Module):
#     def __init__(self):
#         super(vgg16_locate,self).__init__()
#         self.cfg = [512, 512, 512]
#         self.vgg16 = vgg16()
#         # self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         # self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
#         self.layer61 = vgg_ex(self.cfg, 512, 3, 3)
#         self.layer62 = vgg_ex(self.cfg, 512, 6, 6)
#         self.layer63 = vgg_ex(self.cfg, 512, 9, 9)
#         self.layer64 = vgg_ex(self.cfg, 512, 12, 12)
#
#
#         # self.layer6_convert, self.layer6_trans, self.layer6_score = [],[],[]
#         # for ii in range(3):
#         #     self.layer6_convert.append(nn.Conv2d(1024, 512, 3, 1, 1, bias=False))
#         #     self.layer6_trans.append(nn.Conv2d(512, 512, 1, 1, bias=False))
#         #     self.layer6_score.append(nn.Conv2d(512, 1, 1, 1))
#         # self.layer6_convert, self.layer6_trans, self.layer6_score = nn.ModuleList(self.layer6_convert), nn.ModuleList(self.layer6_trans), nn.ModuleList(self.layer6_score)
#         self.trans = nn.Conv2d(512*5, 512, 3, 1, 1, bias=False)
#         # self.score = nn.Conv2d(3, 1, 1, 1)
#         # self.score = nn.Conv2d(1, 1, 1, 1)
#         self.relu = nn.ReLU(inplace=True)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, 0.01)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def load_pretrained_model(self, model):
#         self.vgg16.load_pretrained_model(model)
#
#     def forward(self, x):
#         x_size = x.size()[2:]
#         xs = self.vgg16(x)
#
#         xls = [xs[-1]]
#         xls.append(self.layer61(xs[-2]))
#         xls.append(self.layer62(xs[-2]))
#         xls.append(self.layer63(xs[-2]))
#         xls.append(self.layer64(xs[-2]))
#
#         # xls_tmp = [self.layer6_convert[0](xls[0])]
#         # for ii in range(1, 3):
#         #     xls_tmp.append(F.interpolate(self.layer6_convert[ii](xls[ii]), xls_tmp[0].size()[2:], mode='bilinear', align_corners=True))
#         #
#         # xls_trans = self.layer6_trans[0](xls_tmp[0])
#         # for ii in range(1, 3):
#         #     xls_trans = torch.add(xls_trans, self.layer6_trans[ii](xls_tmp[ii]))
#         score, score_fuse = [], None
#         # for ii in range(3):
#         #     score.append(self.layer6_score[ii](xls_tmp[ii]))
#
#         xls_trans = self.trans(self.relu(torch.cat(xls, dim=1)))
#         xs[-1] = xls_trans
#         # score_fuse = F.interpolate(self.score(torch.cat(score, dim=1)), x_size, mode='bilinear', align_corners=True)
#         # score_fuse = F.interpolate(self.score(torch.add(torch.add(score[0], score[1]), score[2])), x_size, mode='bilinear', align_corners=True)
#
#         # score = [F.interpolate(ss, x_size, mode='bilinear', align_corners=True) for ss in score]
#
#         return xs, score_fuse, score

class vgg16_locate(nn.Module):
    def __init__(self):
        super(vgg16_locate,self).__init__()
        self.vgg16 = vgg16()
        self.in_planes = 512
        # self.out_planes = [512, 256, 128, 64] #  with convert layer, with conv6
        # self.out_planes = [512, 512, 256, 128] #  no convert layer, with conv6
        self.out_planes = [512, 256, 128] #  no convert layer, no conv6

        ppms, infos = [], []
        # for ii in [3, 6, 12]:
        #     if ii <= 8:
        #         ppms.append(nn.Sequential(nn.AvgPool2d(kernel_size=ii, stride=ii), nn.Conv2d(self.in_planes, self.in_planes, 1, 1, bias=False), nn.ReLU(inplace=True)))
        #     else:
        #         ppms.append(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(self.in_planes, self.in_planes, 1, 1, bias=False), nn.ReLU(inplace=True)))
        for ii in [1, 3, 5]:
            ppms.append(nn.Sequential(nn.AdaptiveAvgPool2d(ii), nn.Conv2d(self.in_planes, self.in_planes, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.ppms = nn.ModuleList(ppms)

        self.ppm_cat = nn.Sequential(nn.Conv2d(self.in_planes * 4, self.in_planes, 3, 1, 1, bias=False), nn.ReLU(inplace=True))
        #self.ppm_cat = nn.Sequential(nn.Conv2d(self.in_planes, self.in_planes, 3, 1, 1, bias=False), nn.ReLU(inplace=True))
        # self.ppm_score = nn.Conv2d(self.in_planes, 1, 1, 1)
        for ii in self.out_planes:
            infos.append(nn.Sequential(nn.Conv2d(self.in_planes, ii, 3, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.infos = nn.ModuleList(infos)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_pretrained_model(self, model):
        self.vgg16.load_pretrained_model(model)

    def forward(self, x):
        x_size = x.size()[2:]
        xs = self.vgg16(x)

        xls = [xs[-1]]
        #xls = xs[-1]
        for k in range(len(self.ppms)):
            xls.append(F.interpolate(self.ppms[k](xs[-1]), xs[-1].size()[2:], mode='bilinear', align_corners=True))
            #xls = torch.add(xls, F.interpolate(self.ppms[k](xs[-1]), xs[-1].size()[2:], mode='bilinear', align_corners=True))
        xls = self.ppm_cat(torch.cat(xls, dim=1))
        #xls = self.ppm_cat(xls)
        top_score = None
        # top_score = F.interpolate(self.ppm_score(xls), x_size, mode='bilinear', align_corners=True)

        infos = []
        for k in range(len(self.infos)):
            infos.append(self.infos[k](F.interpolate(xls, xs[len(self.infos) - 1 - k].size()[2:], mode='bilinear', align_corners=True)))

        return xs, top_score, infos

# class vgg16_locate(nn.Module):
#     def __init__(self):
#         super(vgg16_locate,self).__init__()
#         self.cfg = [1024, 1024, 1024]
#         self.vgg16 = vgg16()
#         self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.maxpool6 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.layer5 = vgg_ex(self.cfg, 1024)
#         self.layer6 = vgg_ex(self.cfg, 1024)
#         self.layer7 = vgg_ex(self.cfg, 1024)
#
#         self.layer71 = nn.Conv2d(1024, 512, 1, 1, bias=False)
#         self.layer61 = nn.Conv2d(1024, 512, 1, 1, bias=False)
#         self.layer51 = nn.Conv2d(1024, 512, 1, 1, bias=False)
#         self.layer41 = nn.Conv2d(1024, 512, 1, 1, bias=False)
#
#         self.layer76 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
#         self.layer65 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
#         self.layer54 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 1, 1, bias=False))
#         # self.layer54 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
#         # self.layer54_ = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(512, 512, 1, 1, bias=False))
#         # self.score = nn.Conv2d(512, 1, 1, 1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, 0.01)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def load_pretrained_model(self, model):
#         self.vgg16.load_pretrained_model(model)
#
#     def forward(self, x):
#         x_size = x.size()[2:]
#         score_fuse, score = None, None
#         xs = self.vgg16(x)
#
#         x5 = self.layer5(self.maxpool4(xs[-1]))
#         x6 = self.layer6(self.maxpool5(x5))
#         x7 = self.layer7(self.maxpool6(x6))
#
#         x8 = self.layer76(self.relu(torch.add(F.interpolate(self.layer71(x7) , x6.size()[2:], mode='bilinear', align_corners=True), self.layer61(x6))))
#         x8 = self.layer65(self.relu(torch.add(F.interpolate(x8 , x5.size()[2:], mode='bilinear', align_corners=True), self.layer51(x5))))
#         x8 = self.layer54(self.relu(torch.add(F.interpolate(x8 , xs[-1].size()[2:], mode='bilinear', align_corners=True), self.layer41(xs[-1]))))
#         xs[-1] = x8
#
#         # x8 = self.layer76(self.relu(torch.add(F.interpolate(self.layer71(x7) , x6.size()[2:], mode='bilinear', align_corners=True), self.layer61(x6))))
#         # x9 = self.layer65(self.relu(torch.add(F.interpolate(x8 , x5.size()[2:], mode='bilinear', align_corners=True), self.layer51(x5))))
#         # x10 = self.layer54(self.relu(torch.add(F.interpolate(x9 , xs[-1].size()[2:], mode='bilinear', align_corners=True), self.layer41(xs[-1]))))
#         # score_fuse = F.interpolate(self.score(self.relu(torch.add(torch.add(F.interpolate(x8 , x10.size()[2:], mode='bilinear', align_corners=True),
#         #                        F.interpolate(x9 , x10.size()[2:], mode='bilinear', align_corners=True)), x10))), x_size, mode='bilinear', align_corners=True)
#         # xs[-1] = self.layer54_(x10)
#
#         return xs, score_fuse, score
