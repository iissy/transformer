import math
import torch
from torch import nn


class PositionEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# import plotly.express as px
#
# pe = PositionEncoding(120, 0)
# z = pe.forward(torch.zeros(1, 100, 120))
# df = pd.DataFrame(z[0, :, [0 ,20 ,60 ,110]].data.numpy() ,columns = ["dim " +c for c in ['0' ,'20' ,'60' ,'110']])
# df.insert(0 ,"x" ,np.arange(100))
# px.line(df, x = "x" ,y = ["dim " +c for c in ['0' ,'20' ,'60' ,'110']]).show()
# px.imshow(np.squeeze(z.data.numpy()) ,color_continuous_scale="blues",width=1000,height=800).show()