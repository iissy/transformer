import torch
from torch import nn
import torch.nn.functional as F
import copy
import math


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class ScaledDotProductAttention(nn.Module):
    "Compute 'Scaled Dot Product Attention'"
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e20)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return p_attn @ value, p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # 记录 attention矩阵结果
        self.dropout = nn.Dropout(p=dropout)
        self.attention = ScaledDotProductAttention()

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask,
                                      dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# 为了让训练过程与解码过程信息流一致，遮挡tgt序列后面元素，设置其注意力为0
def tril_mask(data):
    size = data.size(-1)
    full = torch.full((1, size, size), 1, dtype=torch.int, device=data.device)
    mask = torch.tril(full).bool()
    return mask


# 设置对<PAD>的注意力为0
def pad_mask(data, pad=0):
    "Mask out pad positions."
    mask = (data != pad).unsqueeze(-2)
    return mask


# 计算一个batch数据的src_mask和tgt_mask
class MaskedBatch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = pad_mask(src, pad)
        if tgt is not None:
            self.tgt = tgt[:, :-1]  # 训练时,拿tgt的每一个词输入,去预测下一个词,所以最后一个词无需输入
            self.tgt_y = tgt[:, 1:]  # 第一个总是<SOS>无需预测，预测从第二个词开始
            self.tgt_mask = self.make_tgt_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).sum()

    @staticmethod
    def make_tgt_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_pad_mask = pad_mask(tgt, pad)
        tgt_tril_mask = tril_mask(tgt)
        tgt_mask = tgt_pad_mask & tgt_tril_mask
        return tgt_mask


# import plotly.express as px

# 测试tril_mask
# data = torch.zeros(3,5)
# mask = tril_mask(data)
# f = px.imshow(mask[0], color_continuous_scale="blues", height=600, width=600)
# f.show()


# query = torch.tensor([[[0.0,1.414],[1.414,0.0],[1.0,1.0],[-1.0,1.0],[1.0,-1.0]]])
# key = query.clone()
# value = query.clone()
# attention = ScaledDotProductAttention()
#
# #没有mask
# out, p_att = attention(query, key, value)
# print(out)
# print(p_att)
# fig = px.imshow(p_att[0], color_continuous_scale="blues", title="without mask", height=600, width=600)
# fig.show()
#
# #考虑mask
# out, p_att = attention(query, key, value, mask = tril_mask(torch.zeros(3,5)))
# print(out)
# print(p_att)
# fig = px.imshow(p_att[0], color_continuous_scale="blues", height=600, width=600, title="with mask")
# fig.show()


# 测试MultiHeadAttention
# cross_attn = MultiHeadAttention(h=2, d_model=4)
# cross_attn.eval()
# q1 = torch.tensor([[[0.1,0.1,0.1,0.1],[0.1,0.3,0.1,0.3]]])
# k1 = q1.clone()
# v1 = q1.clone()
# tgt_mask = tril_mask(torch.zeros(2,2))
#
# out1 = cross_attn.forward(q1,k1,v1,mask = tgt_mask)
# print("out1:\n",out1)

#改变序列的第2个元素取值，由于有mask的遮挡，不会影响第1个输出
# q2 = torch.tensor([[[0.1,0.1,0.1,0.1],[0.4,0.5,0.5,0.8]]])
# k2 = q2.clone()
# v2 = q2.clone()
# tgt_mask = tril_mask(torch.zeros(2,2))
# out2 = cross_attn.forward(q2,k2,v2,mask = tgt_mask)
# print("out2:\n",out2)


# 测试MaskedBatch
# for src, tgt in dl_train:
#     mbatch = MaskedBatch(src = src,tgt = tgt, pad = 0)
#     print(mbatch.src.shape)
#     print(mbatch.tgt.shape)
#     print(mbatch.tgt_y.shape)
#
#     print(mbatch.src_mask.shape)
#     print(mbatch.tgt_mask.shape)
#     fig = px.imshow(mbatch.tgt_mask[0],color_continuous_scale="blues",width=600,height=600)
#     fig.show()
#     break

