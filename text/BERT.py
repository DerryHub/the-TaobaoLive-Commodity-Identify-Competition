import torch
from torch import nn
from bert.bert import BertModel
from arcface.utils import l2_norm

class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        config.output_size = config.embedding_size
        embedding_size = config.embedding_size
        drop_ratio = config.drop_ratio
        self.bert = BertModel(config)
        self.output_layer = nn.Sequential(
                                nn.BatchNorm1d(embedding_size),
                                nn.Dropout(drop_ratio),
                                nn.Linear(embedding_size, embedding_size),
                                nn.BatchNorm1d(embedding_size))

    def forward(self, text):
        output = self.bert(text)
        output = self.output_layer(output)

        return l2_norm(output)