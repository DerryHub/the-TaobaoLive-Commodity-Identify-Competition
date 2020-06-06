from arcface.cpn.resnet import ResNet
import torch.nn as nn
import torch
from arcface.cpn.globalNet import globalNet
from arcface.cpn.refineNet import refineNet
from arcface.utils import Flatten, l2_norm

class CPN(nn.Module):
    def __init__(self, config):
        super(CPN, self).__init__()
        # self.output_shape = config.output_shape
        self.output_shape = (8, 8)
        self.num_class = 256
        # self.num_class = config.num_labels
        self.resnet = ResNet(config)
        #self.resnet = resnet50(config)
        channel_settings = [2048, 1024, 512, 256]
        # self.embedding_size = 2048
        # drop_ratio = 0.1
        self.embedding_size = config.embedding_size
        drop_ratio = config.drop_ratio

        self.global_net = globalNet(channel_settings, self.output_shape, self.num_class)

        #self.refine_net = refineNet(channel_settings[-1], self.output_shape, self.num_class)
        self.output_layer = nn.Sequential(
                                    nn.Conv2d(self.num_class * 4, 2048, 3, 1),
                                    nn.BatchNorm2d(2048),
                                    nn.AdaptiveAvgPool2d((1, 1)),
                                    Flatten(),
                                    nn.Linear(2048, self.embedding_size),
                                    nn.BatchNorm1d(self.embedding_size)
                                    )
                                    

    def forward(self, x):
        #print(x.size())
        res_out = self.resnet(x)
        # print(res_out[0].size(),res_out[1].size(),res_out[2].size(),res_out[3].size())
        
        #global_fms,global_outs = self.global_net(res_out[0],res_out[1],res_out[2],res_out[3]) global_fms10,global_fms11,global_fms12,global_fms13,
        g_output10,g_output11,g_output12,g_output13= self.global_net(res_out[0],res_out[1],res_out[2],res_out[3])
        #print('after output')
        # 
        #global_fms = [global_fms10,global_fms11,global_fms12,global_fms13]
        #refine_out = self.refine_net( global_fms10,global_fms11,global_fms12,global_fms13)
        total = [g_output10,g_output11,g_output12,g_output13]

        out = torch.cat(total, dim=1)
        # print(g_output10.size(),g_output11.size(),g_output12.size(),g_output13.size())
        #print('out size :')
        #print(out.size())
        output = self.output_layer(out)
        #return l2_norm(x)

       

        return l2_norm(output)#g_output10,g_output11,g_output12,g_output13 , refine_out


if __name__ == "__main__":
    net = CPN('aa')
    a = torch.randn([5, 3, 224, 224])
    b = net(a)
    print(b.size())