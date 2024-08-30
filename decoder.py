from torch import nn
from attention import clones,MultiHeadAttention
from add import ResConnection
from norm import LayerNorm
from feedforward import FeedForward

class TransformerDecoderLayer(nn.Module):
    "TransformerDecoderLayer is made of self-attn, cross-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, cross_attn, feed_forward, dropout):
        super(TransformerDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.res_layers = clones(ResConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        x = self.res_layers[0](x, lambda m: self.self_attn(m, m, m, tgt_mask))
        x = self.res_layers[1](x, lambda m: self.cross_attn(m, memory, memory, src_mask))
        return self.res_layers[2](x, self.feed_forward)


class TransformerDecoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(TransformerDecoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

    @classmethod
    def from_config(cls ,N=6 ,d_model=512, d_ff=2048, h=8, dropout=0.1):
        self_attn = MultiHeadAttention(h, d_model)
        cross_attn = MultiHeadAttention(h, d_model)
        ff = FeedForward(d_model, d_ff, dropout)
        layer = TransformerDecoderLayer(d_model, self_attn, cross_attn, ff, dropout)
        return cls(layer ,N)


# from torchkeras import summary
#
# mbatch = MaskedBatch(src=src ,tgt=tgt ,pad=0)
#
# src_embed = nn.Sequential(WordEmbedding(d_model=32, vocab = len(vocab_x)),
#                           PositionEncoding(d_model=32, dropout=0.1))
# encoder = TransformerEncoder.from_config(N=3 ,d_model=32, d_ff=128, h=8, dropout=0.1)
# memory = encoder(src_embed(src) ,mbatch.src_mask)
#
# tgt_embed = nn.Sequential(WordEmbedding(d_model=32, vocab = len(vocab_y)),
#                           PositionEncoding(d_model=32, dropout=0.1))
# decoder = TransformerDecoder.from_config(N=3 ,d_model=32, d_ff=128, h=8, dropout=0.1)
#
# result = decoder.forward(tgt_embed(mbatch.tgt) ,memory ,mbatch.src_mask ,mbatch.tgt_mask)
# summary(decoder ,input_data_args = [tgt_embed(mbatch.tgt) ,memory,
#                                    mbatch.src_mask ,mbatch.tgt_mask]);
#
#
# decoder.eval()
# mbatch.tgt[0][1 ] =8
# result = decoder.forward(tgt_embed(mbatch.tgt) ,memory ,mbatch.src_mask ,mbatch.tgt_mask)
# print(torch.sum(result[0][0]))
#
# mbatch.tgt[0][1 ] =7
# result = decoder.forward(tgt_embed(mbatch.tgt) ,memory ,mbatch.src_mask ,mbatch.tgt_mask)
# print(torch.sum(result[0][0]))