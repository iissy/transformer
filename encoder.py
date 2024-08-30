from torch import nn
from attention import clones,MultiHeadAttention,pad_mask
from add import ResConnection
from norm import LayerNorm
from feedforward import FeedForward

class TransformerEncoderLayer(nn.Module):
    "TransformerEncoderLayer is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.res_layers = clones(ResConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.res_layers[0](x, lambda m: self.self_attn(m, m, m, mask))
        return self.res_layers[1](x, self.feed_forward)


class TransformerEncoder(nn.Module):
    "TransformerEncoder is a stack of N TransformerEncoderLayer"
    def __init__(self, layer, N):
        super(TransformerEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    @classmethod
    def from_config(cls ,N=6 ,d_model=512, d_ff=2048, h=8, dropout=0.1):
        attn = MultiHeadAttention(h, d_model)
        ff = FeedForward(d_model, d_ff, dropout)
        layer = TransformerEncoderLayer(d_model, attn, ff, dropout)
        return cls(layer ,N)


