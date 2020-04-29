import torch
import torch.nn as nn
import torch.nn.functional as F
from arcface.utils import l2_norm


class TextCNN(nn.Module):
    
    def __init__(self, config):
        super(TextCNN, self).__init__()
        #load pretrained embedding in embedding layer.
        embedding_size = config.embedding_size
        vocab_size = config.vocab_size

        # embedding_size = 2048
        # vocab_size = 100
        kernel_wins = [3,4,5]
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
    
        #Convolutional Layers with different window size kernels
        self.convs = nn.ModuleList([nn.Conv2d(1, 512, (w, embedding_size)) for w in kernel_wins])
        #Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        #FC layer
        self.fc = nn.Linear(len(kernel_wins)*512, embedding_size)
        
    def forward(self, x):
        emb_x = self.embed(x)
        emb_x = emb_x.unsqueeze(1)

        con_x = [conv(emb_x) for conv in self.convs]

        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
        
        fc_x = torch.cat(pool_x, dim=1)
        
        fc_x = fc_x.squeeze(-1)

        fc_x = self.dropout(fc_x)
        logit = self.fc(fc_x)
        return l2_norm(logit)

if __name__ == "__main__":
    net = TextCNN('')
    a = torch.zeros([4,64]).long()
    b = net(a)
    print(b.size())