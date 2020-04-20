import torch
from torch import nn
from TFIDF.tf_idf import SentVec_TFIDF
from arcface.utils import l2_norm

class TF_IDF(nn.Module):
    def __init__(self, config):
        super(TF_IDF, self).__init__()
        embedding_size = config.embedding_size
        drop_ratio = config.drop_ratio

        self.ti = SentVec_TFIDF(embedding_size)

        self.output_layer = nn.Sequential(
                                nn.BatchNorm1d(embedding_size),
                                nn.Dropout(drop_ratio),
                                nn.Linear(embedding_size, embedding_size),
                                nn.BatchNorm1d(embedding_size))
        
    def forward(self, text):
        embeddings = self.ti(text)
        out = self.output_layer(embeddings)
        return l2_norm(out)
