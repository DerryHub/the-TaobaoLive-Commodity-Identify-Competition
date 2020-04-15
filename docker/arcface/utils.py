import os
import json
import torch
from torch import nn

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SentVec_TFIDF(nn.Module):
    def __init__(self, embedding_size, root_dir='data'):
        super(SentVec_TFIDF, self).__init__()
        with open(os.path.join(root_dir, 'TF_IDF.json'), 'r') as f:
            TI_dic = json.load(f)
        max_size = len(TI_dic)
        self.TI = torch.zeros(max_size).float()
        for k in TI_dic.keys():
            self.TI[int(k)] = TI_dic[k]
        self.embedding = nn.Embedding(len(self.TI), embedding_size, padding_idx=0)
        
    def forward(self, words):
        embeddings = self.embedding(words)
        weight = self.TI[words].to(words.device)
        weight /= (weight.sum(dim=1).view(-1, 1)+1e-8)
        embeddings *= weight.view(-1, words.size(1), 1)
        sentEmbeddings = embeddings.sum(dim=1)
        return sentEmbeddings

class IorV(nn.Module):
    def __init__(self, embedding_size):
        super(IorV, self).__init__()
        self.embedding = nn.Embedding(2, embedding_size)
    
    def forward(self, x):
        return self.embedding(x)