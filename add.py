from torch import nn
from norm import LayerNorm

class ResConnection(nn.Module):
    """
    A residual connection with a layer norm.
    Note the norm is at last according to the paper, but it may be better at first.
    """
    def __init__(self, size, dropout, norm_first=True):
        super(ResConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if self.norm_first:
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            return self.norm(x + self.dropout(sublayer(x)))
