import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):  # size为词典大小
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))  # 预测结果不会是<SOS> #和<PAD>
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero((target.data == self.padding_idx).int())
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


# import plotly.express as px
#
# # Example of label smoothing.
# smooth_loss = LabelSmoothingLoss(5, 0, 0.4)
# predict = torch.FloatTensor([[1e-10, 0.2, 0.7, 0.1, 1e-10],
#                              [1e-10, 0.2, 0.7, 0.1, 1e-10],
#                              [1e-10, 0.2, 0.7, 0.1, 1e-10]])
# loss = smooth_loss(predict.log(), torch.LongTensor([2, 1, 0]))
#
# print("smoothed target:\n", smooth_loss.true_dist, "\n")
# print("loss:", loss)
# px.imshow(smooth_loss.true_dist, color_continuous_scale="blues", height=600, width=1000).show()
