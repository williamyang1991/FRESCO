import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np

from resnet import resnet50
from vgg import vgg16


config_vgg = {'convert': [[128,256,512,512,512],[64,128,256,512,512]], 'merge1': [[128, 256, 128, 3,1], [256, 512, 256, 3, 1], [512, 0, 512, 5, 2], [512, 0, 512, 5, 2],[512, 0, 512, 7, 3]], 'merge2': [[128], [256, 512, 512, 512]]}  # no convert layer, no conv6

config_resnet = {'convert': [[64,256,512,1024,2048],[128,256,512,512,512]], 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False], [True, True, True, True, False]], 'score': 256, 'edgeinfo':[[16, 16, 16, 16], 128, [16,8,4,2]],'edgeinfoc':[64,128], 'block': [[512, [16]], [256, [16]], [256, [16]], [128, [16]]], 'fuse': [[16, 16, 16, 16], True], 'fuse_ratio': [[16,1], [8,1], [4,1], [2,1]],  'merge1': [[128, 256, 128, 3,1], [256, 512, 256, 3, 1], [512, 0, 512, 5, 2], [512, 0, 512, 5, 2],[512, 0, 512, 7, 3]], 'merge2': [[128], [256, 512, 512, 512]]}


class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        up0, up1, up2 = [], [], []
        for i in range(len(list_k[0])):
          
            up0.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.ReLU(inplace=True)))


        self.convert0 = nn.ModuleList(up0)


    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl


        

class MergeLayer1(nn.Module): # list_k: [[64, 512, 64], [128, 512, 128], [256, 0, 256] ... ]
    def __init__(self, list_k):
        super(MergeLayer1, self).__init__()
        self.list_k = list_k
        trans, up, score = [], [], []
        for ik in list_k:
            if ik[1] > 0:
                trans.append(nn.Sequential(nn.Conv2d(ik[1], ik[0], 1, 1, bias=False), nn.ReLU(inplace=True)))

           
            up.append(nn.Sequential(nn.Conv2d(ik[0], ik[2], ik[3], 1, ik[4]), nn.ReLU(inplace=True), nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]), nn.ReLU(inplace=True), nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]), nn.ReLU(inplace=True)))
            score.append(nn.Conv2d(ik[2], 1, 3, 1, 1))
        trans.append(nn.Sequential(nn.Conv2d(512, 128, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.trans, self.up, self.score = nn.ModuleList(trans), nn.ModuleList(up), nn.ModuleList(score)
        self.relu =nn.ReLU()

    def forward(self, list_x, x_size):
        up_edge, up_sal, edge_feature, sal_feature = [], [], [], []
        
        
        num_f = len(list_x)
        tmp = self.up[num_f - 1](list_x[num_f-1])
        sal_feature.append(tmp)
        U_tmp = tmp
        up_sal.append(F.interpolate(self.score[num_f - 1](tmp), x_size, mode='bilinear', align_corners=True))
        
        for j in range(2, num_f ):
            i = num_f - j
             
            if list_x[i].size()[1] < U_tmp.size()[1]:
                U_tmp = list_x[i] + F.interpolate((self.trans[i](U_tmp)), list_x[i].size()[2:], mode='bilinear', align_corners=True)
            else:
                U_tmp = list_x[i] + F.interpolate((U_tmp), list_x[i].size()[2:], mode='bilinear', align_corners=True)
            
            
            
                
            
            tmp = self.up[i](U_tmp)
            U_tmp = tmp
            sal_feature.append(tmp)
            up_sal.append(F.interpolate(self.score[i](tmp), x_size, mode='bilinear', align_corners=True))

        U_tmp = list_x[0] + F.interpolate((self.trans[-1](sal_feature[0])), list_x[0].size()[2:], mode='bilinear', align_corners=True)
        tmp = self.up[0](U_tmp)
        edge_feature.append(tmp)
       
        up_edge.append(F.interpolate(self.score[0](tmp), x_size, mode='bilinear', align_corners=True)) 
        return up_edge, edge_feature, up_sal, sal_feature        
        
class MergeLayer2(nn.Module): 
    def __init__(self, list_k):
        super(MergeLayer2, self).__init__()
        self.list_k = list_k
        trans, up, score = [], [], []
        for i in list_k[0]:
            tmp = []
            tmp_up = []
            tmp_score = []
            feature_k = [[3,1],[5,2], [5,2], [7,3]]
            for idx, j in enumerate(list_k[1]):
                tmp.append(nn.Sequential(nn.Conv2d(j, i, 1, 1, bias=False), nn.ReLU(inplace=True)))

                tmp_up.append(nn.Sequential(nn.Conv2d(i , i, feature_k[idx][0], 1, feature_k[idx][1]), nn.ReLU(inplace=True), nn.Conv2d(i, i,  feature_k[idx][0],1 , feature_k[idx][1]), nn.ReLU(inplace=True), nn.Conv2d(i, i, feature_k[idx][0], 1, feature_k[idx][1]), nn.ReLU(inplace=True)))
                tmp_score.append(nn.Conv2d(i, 1, 3, 1, 1))
            trans.append(nn.ModuleList(tmp))

            up.append(nn.ModuleList(tmp_up))
            score.append(nn.ModuleList(tmp_score))
            

        self.trans, self.up, self.score = nn.ModuleList(trans), nn.ModuleList(up), nn.ModuleList(score)       
        self.final_score = nn.Sequential(nn.Conv2d(list_k[0][0], list_k[0][0], 5, 1, 2), nn.ReLU(inplace=True), nn.Conv2d(list_k[0][0], 1, 3, 1, 1))
        self.relu =nn.ReLU()

    def forward(self, list_x, list_y, x_size):
        up_score, tmp_feature = [], []
        list_y = list_y[::-1]

        
        for i, i_x in enumerate(list_x):
            for j, j_x in enumerate(list_y):                              
                tmp = F.interpolate(self.trans[i][j](j_x), i_x.size()[2:], mode='bilinear', align_corners=True) + i_x                
                tmp_f = self.up[i][j](tmp)             
                up_score.append(F.interpolate(self.score[i][j](tmp_f), x_size, mode='bilinear', align_corners=True))                  
                tmp_feature.append(tmp_f)
       
        tmp_fea = tmp_feature[0]
        for i_fea in range(len(tmp_feature) - 1):
            tmp_fea = self.relu(torch.add(tmp_fea, F.interpolate((tmp_feature[i_fea+1]), tmp_feature[0].size()[2:], mode='bilinear', align_corners=True)))
        up_score.append(F.interpolate(self.final_score(tmp_fea), x_size, mode='bilinear', align_corners=True))
      


        return up_score
       


# extra part
def extra_layer(base_model_cfg, vgg):
    if base_model_cfg == 'vgg':
        config = config_vgg
    elif base_model_cfg == 'resnet':
        config = config_resnet
    merge1_layers = MergeLayer1(config['merge1'])
    merge2_layers = MergeLayer2(config['merge2'])

    return vgg, merge1_layers, merge2_layers


# TUN network
class TUN_bone(nn.Module):
    def __init__(self, base_model_cfg, base, merge1_layers, merge2_layers):
        super(TUN_bone, self).__init__()
        self.base_model_cfg = base_model_cfg
        if self.base_model_cfg == 'vgg':

            self.base = base
            # self.base_ex = nn.ModuleList(base_ex)
            self.merge1 = merge1_layers
            self.merge2 = merge2_layers

        elif self.base_model_cfg == 'resnet':
            self.convert = ConvertLayer(config_resnet['convert'])
            self.base = base
            self.merge1 = merge1_layers
            self.merge2 = merge2_layers

    def forward(self, x):
        x_size = x.size()[2:]
        conv2merge = self.base(x)        
        if self.base_model_cfg == 'resnet':            
            conv2merge = self.convert(conv2merge)
        up_edge, edge_feature, up_sal, sal_feature = self.merge1(conv2merge, x_size)
        up_sal_final = self.merge2(edge_feature, sal_feature, x_size)
        return up_edge, up_sal, up_sal_final


# build the whole network
def build_model(base_model_cfg='vgg'):
    if base_model_cfg == 'vgg':
        return TUN_bone(base_model_cfg, *extra_layer(base_model_cfg, vgg16()))
    elif base_model_cfg == 'resnet':
        return TUN_bone(base_model_cfg, *extra_layer(base_model_cfg, resnet50()))


# weight init
def xavier(param):
    # init.xavier_uniform(param)
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()

if __name__ == '__main__':
    from torch.autograd import Variable
    net = TUN(*extra_layer(vgg(base['tun'], 3), vgg(base['tun_ex'], 512), config['merge_block'], config['fuse'])).cuda()
    img = Variable(torch.randn((1, 3, 256, 256))).cuda()
    out = net(img, mode = 2)
    print(len(out))
    print(len(out[0]))
    print(out[0].shape)
    print(len(out[1]))
    # print(net)
    input('Press Any to Continue...')
