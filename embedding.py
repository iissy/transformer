import math
from torch import nn

# 单词嵌入
class WordEmbedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) #note here, multiply sqrt(d_model)
