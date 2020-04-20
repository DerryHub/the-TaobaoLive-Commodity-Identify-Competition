import os
import torch
from torch import nn
import json

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

if __name__ == "__main__":
    model = SentVec_TFIDF(2048)
    a = torch.zeros([4, 64]).long()
    b = model(a)
    print(b.size())