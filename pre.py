import torch
from attention import MaskedBatch,tril_mask
from transformer import Transformer
from labelsmoothing import LabelSmoothingLoss
from optim import NoamOpt
from dataset import vocab_x,vocab_y,vocab_xr,vocab_yr
from dataset import dl_train,dl_val

model = Transformer.from_config(src_vocab=len(vocab_x), tgt_vocab=len(vocab_y), N=5, d_model=64, d_ff=128, h=8, dropout=0.1)
loss_fn = LabelSmoothingLoss(size=len(vocab_y), padding_idx=0, smoothing=0.1)
optimizer = NoamOpt(model.parameters(), model_size=64)
model.to("cuda")

def train():
    for epoch in range(10):
        model.train()
        for step, data in enumerate(dl_train):
            batch = MaskedBatch(src=data[0], tgt=data[1], pad=0)
            out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            targets = out.reshape(-1, out.size(-1))
            labels = batch.tgt_y.reshape(-1)
            loss = loss_fn(targets, labels) / batch.ntokens
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 50 == 0:
                print(epoch, loss.item())

        y_pred = greedy_decode(model, batch.src, batch.src_mask, 50, vocab_y["<SOS>"])
        print("input:")
        print(get_words(batch.src[0], vocab_xr), '\n')  # 标签结果
        print("ground truth:")
        print(get_words(batch.tgt[0], vocab_yr), '\n')  # 标签结果
        print("prediction:")
        print(get_words(y_pred[0], vocab_yr))  # 解码预测结果，原始标签中<PAD>位置的预测可以忽略

def get_raw_words(tensor,vocab_r) ->"str":
    words = [vocab_r[i] for i in tensor.tolist()]
    return words

def get_words(tensor, vocab_r) ->"str":
    s = "".join([vocab_r[i] for i in tensor.tolist()])
    words = s[:s.find('<EOS>')].replace('<SOS>','')
    return words

def prepare(x):
    return x.to("cuda")

def greedy_decode(net, src, src_mask, max_len, start_symbol):
    net.eval()
    memory = net.encode(src, src_mask)
    ys = torch.full((len(src), max_len), start_symbol, dtype = src.dtype).to(src.device)
    for i in range(max_len-1):
        out = net.generator(net.decode(memory, src_mask, ys, tril_mask(ys)))
        ys[:,i+1]=out.argmax(dim=-1)[:,i]
    return ys


if __name__ == '__main__':
    train()