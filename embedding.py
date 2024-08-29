import math
from torch import nn

# 单词嵌入
class WordEmbedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # note here, multiply sqrt(d_model)
        # 目的是加大输出的值，避免嵌入信息丢失
        return self.embedding(x) * math.sqrt(self.d_model)
