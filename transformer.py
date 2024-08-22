from torch import nn
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from embedding import WordEmbedding
from position import PositionEncoding
from generator import Generator


class Transformer(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """
    def __init__(self, encoder, de, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = de
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.reset_parameters()

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.generator(self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask))

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    @classmethod
    def from_config(cls, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        enc = TransformerEncoder.from_config(N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)
        dec = TransformerDecoder.from_config(N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)
        src_embed = nn.Sequential(WordEmbedding(d_model, src_vocab), PositionEncoding(d_model, dropout))
        tgt_embed = nn.Sequential(WordEmbedding(d_model, tgt_vocab), PositionEncoding(d_model, dropout))

        generator = Generator(d_model, tgt_vocab)
        return cls(enc, dec, src_embed, tgt_embed, generator)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


# from torchkeras import summary
#
# for src, tgt in dl_train:
#     print(src.size(), tgt.size())
#     net = Transformer.from_config(src_vocab=len(vocab_x), tgt_vocab=len(vocab_y), N=2, d_model=32, d_ff=128, h=8, dropout=0.1)
#     mbatch = MaskedBatch(src=src, tgt=tgt, pad=0)
#     print(mbatch)
#     summary(net, input_data_args=[mbatch.src, mbatch.tgt, mbatch.src_mask, mbatch.tgt_mask])
#     break
