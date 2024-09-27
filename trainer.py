import torch
import time
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
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
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

            total_loss += loss.item()
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            if step % 50 == 1:
                elapsed = time.time() - start
                print("Epoch Step: %d/%d Loss: %f Tokens per Sec: %f" % (step, epoch, loss, tokens / elapsed))
                start = time.time()
                tokens = 0

    torch.save(model.state_dict(), "checkpoint.pth")

if __name__ == '__main__':
    train()