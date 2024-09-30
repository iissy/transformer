import torch
import time
import numpy as np
from accelerate import Accelerator
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from attention import MaskedBatch
from transformer import Transformer
from labelsmoothing import LabelSmoothingLoss
from optim import NoamOpt
from dataset import vocab_x,vocab_y
from dataset import dl_train
import config

model = Transformer.from_config(src_vocab=len(vocab_x), tgt_vocab=len(vocab_y), N=5, d_model=64, d_ff=128, h=8, dropout=0.1)
loss_fn = LabelSmoothingLoss(size=len(vocab_y), padding_idx=0, smoothing=0.1)
optimizer = NoamOpt(model.parameters(), model_size=64)
accelerator = Accelerator()
model, optimizer, opt_data = accelerator.prepare(model, optimizer, dl_train)

def train():
    start = time.time()
    global_start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    epochs = 15
    steps_per_epoch = np.floor(config.train_data_size / config.train_batch_size) if config.drop_last else np.ceil(config.train_data_size / config.train_batch_size)
    logging_steps = 50
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        TextColumn("[bold blue]{task.fields[show_info]}"),
        refresh_per_second=1)

    epoch_progress = progress.add_task(description='epoch: ', show_info='epoch: 0/{}'.format(epochs), total=epochs)
    steps_progress = progress.add_task(description='steps: ', show_info='', total=np.ceil(steps_per_epoch / logging_steps))
    progress.start()

    for epoch in range(epochs):
        progress.reset(steps_progress)
        model.train()
        for step, data in enumerate(opt_data, 1):
            src, tgt = data
            batch = MaskedBatch(src=src, tgt=tgt, pad=0)
            out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            targets = out.reshape(-1, out.size(-1))
            labels = batch.tgt_y.reshape(-1)
            loss = loss_fn(targets, labels) / batch.ntokens
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            if step % logging_steps == 0 or step == steps_per_epoch:
                elapsed = time.time() - start
                step_show_txt = 'step: {}/{}, loss: {:.6f}, tokens/sec: {:.0f}'.format(step, steps_per_epoch, loss, tokens / elapsed)
                progress.advance(steps_progress, advance=1)
                progress.update(steps_progress, show_info=step_show_txt)
                start = time.time()
                tokens = 0

        epoch_show_txt = 'epoch: {}/{}'.format(epoch+1, epochs)
        progress.advance(epoch_progress, advance=1)
        progress.update(epoch_progress, show_info=epoch_show_txt)

    progress.refresh()
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    model_dict = accelerator.get_state_dict(unwrapped_model)
    torch.save(model_dict, "checkpoint.pth")

    spend = time.time() - global_start
    print("total time: %d" % spend)


if __name__ == '__main__':
    train()