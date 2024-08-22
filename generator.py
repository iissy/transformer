from torch import nn
import torch.nn.functional as F

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# generator = Generator(d_model = 32, vocab = len(vocab_y))
# log_probs  = generator(result)
# probs = torch.exp(log_probs)
# print("output_probs.shape:",probs.shape)
# print("sum(probs)=1:")
# print(torch.sum(probs,dim = -1)[0])
#
# summary(generator,input_data = result)
