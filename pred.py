import torch
from attention import tril_mask
from dataset import get_data,vocab_x,vocab_y,vocab_xr,vocab_yr
from attention import MaskedBatch
from transformer import Transformer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def greedy_decode(net, src, src_mask, max_len, start_symbol):
    net.eval()
    memory = net.encode(src, src_mask)
    ys = torch.full((len(src), max_len), start_symbol, dtype = src.dtype).to(src.device)
    for i in range(max_len-1):
        out = net.generator(net.decode(memory, src_mask, ys, tril_mask(ys)))
        ys[:,i+1]=out.argmax(dim=-1)[:,i]
    return ys

def get_raw_words(tensor,vocab_r) ->"str":
    words = [vocab_r[i] for i in tensor.tolist()]
    return words

def get_words(tensor, vocab_r) ->"str":
    s = "".join([vocab_r[i] for i in tensor.tolist()])
    words = s[:s.find('<EOS>')].replace('<SOS>','')
    return words


##解码翻译结果
net = Transformer.from_config(src_vocab=len(vocab_x), tgt_vocab=len(vocab_y), N=5, d_model=64, d_ff=128, h=8, dropout=0.1)
net.load_state_dict(torch.load("checkpoint.pth", map_location=device))
net = net.to(device)
src, tgt = get_data()
src, tgt = src.to(device), tgt.to(device)
masked = MaskedBatch(src=src.unsqueeze(dim=0), tgt=tgt.unsqueeze(dim=0))
y_pred = greedy_decode(net, masked.src, masked.src_mask, 50, vocab_y["<SOS>"])
print("input:")
print(get_words(masked.src[0], vocab_xr),'\n') #标签结果
print("ground truth:")
print(get_words(masked.tgt[0], vocab_yr),'\n') #标签结果
print("prediction:")
print(get_words(y_pred[0], vocab_yr)) #解码预测结果，原始标签中<PAD>位置的预测可以忽略