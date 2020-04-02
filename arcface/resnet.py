import torch
from torch import nn
from collections import namedtuple
from utils import l2_norm, Flatten
from bert.bert import BertModel


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride ,bias=False), nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1 ,bias=False), nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1 ,bias=False), nn.BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride ,bias=False), 
                nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3,3), (1,1),1 ,bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3,3), stride, 1 ,bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth,16)
            )
    def forward(self,x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''
    
def get_block(in_channel, depth, num_units, stride = 2):
  return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units-1)]

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units = 3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks

class ResNet(nn.Module):
    def __init__(self, config):
        super(ResNet, self).__init__()
        num_layers = config.num_layers_r
        drop_ratio = config.drop_ratio
        mode = config.mode
        embedding_size = config.embedding_size
        self.sentvec = BertModel(config)

        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1 ,bias=False), 
                                      nn.BatchNorm2d(64), 
                                      nn.PReLU(64))
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512), 
                                       nn.Dropout(drop_ratio),
                                       Flatten(),
                                       nn.Linear(512 * 7 * 7, embedding_size),
                                       nn.BatchNorm1d(embedding_size))
        
        self.last_layer = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = nn.Sequential(*modules)

        if not config.resume:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, text):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        sent = self.sentvec(text)
        x = torch.cat((x, sent), dim=1)
        x = self.last_layer(x)
        return l2_norm(x)


if __name__ == "__main__":
    import argparse

    def get_args_arcface():
        parser = argparse.ArgumentParser("ArcFace")
        parser.add_argument("--size", type=int, default=112, help="The common width and height for all images")
        parser.add_argument("--batch_size", type=int, default=60, help="The number of images per batch")
        parser.add_argument("--lr", type=float, default=1e-6)
        parser.add_argument("--num_epochs", type=int, default=500)
        parser.add_argument("--data_path", type=str, default="data", help="the root folder of dataset")
        parser.add_argument("--saved_path", type=str, default="trained_models")
        parser.add_argument("--num_classes", type=int, default=None)
        parser.add_argument("--num_labels", type=int, default=None)
        parser.add_argument("--drop_ratio", type=float, default=0.1)
        parser.add_argument("--embedding_size", type=int, default=512)
        parser.add_argument('--resume', type=bool, default=True)
        parser.add_argument("--workers", type=int, default=24)
        parser.add_argument('--pretrain', type=bool, default=False)
        parser.add_argument("--s", type=float, default=64.0)
        parser.add_argument("--m", type=float, default=0.5)
        parser.add_argument('--alpha', type=float, default=0.25)
        parser.add_argument('--gamma', type=float, default=1.5)
        parser.add_argument('--threshold', type=float, default=0.6)
        parser.add_argument("--GPUs", type=list, default=[0])
        parser.add_argument("--n_samples", type=int, default=4)
        parser.add_argument("--network", type=str, default='densenet', 
                            help="[resnet, googlenet, inceptionv4, inceptionresnetv2, densenet]")

        # resnet config
        parser.add_argument("--num_layers_r", type=int, default=50, help="[50, 100, 152]")
        parser.add_argument("--mode", type=str, default='ir_se', help="[ir, ir_se]")

        # googlenet config

        # inceptionv4 config

        # inceptionresnetv2 config

        # densenet config
        parser.add_argument("--num_layers_d", type=int, default=121, help="[121, 161, 169, 201]")

        # BERT config
        parser.add_argument("--vocab_size", type=int, default=100)
        parser.add_argument("--hidden_size", type=int, default=256)
        parser.add_argument("--num_hidden_layers", type=int, default=4)
        parser.add_argument("--num_attention_heads", type=int, default=4)
        parser.add_argument("--intermediate_size", type=int, default=512)
        parser.add_argument("--hidden_act", type=str, default='gelu')
        parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
        parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)
        parser.add_argument("--max_position_embeddings", type=int, default=64)
        parser.add_argument("--initializer_range", type=float, default=0.02)
        parser.add_argument("--layer_norm_eps", type=int, default=1e-12)
        parser.add_argument("--output_size", type=int, default=512)
        parser.add_argument("--PAD", type=int, default=0)
        
        args = parser.parse_args()
        return args
    config = get_args_arcface()
    a = torch.randn([2,3,112,112])
    b = torch.Tensor([[0]*64]*2).long()
    net = ResNet(config)
    x = net(a,b)
    print(x.size())