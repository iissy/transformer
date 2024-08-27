from torchkeras import KerasModel
from attention import MaskedBatch
from transformer import Transformer
from labelsmoothing import LabelSmoothingLoss
from optim import NoamOpt
from dataset import vocab_x,vocab_y
from dataset import dl_train,dl_val

class StepRunner:
    def __init__(self, net, loss_fn, accelerator=None, stage="train", metrics_dict=None, optimizer=None, lr_scheduler=None):
        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.accelerator = accelerator
        if self.stage == 'train':
            self.net.train()
        else:
            self.net.eval()

    def __call__(self, batch):
        src, tgt = batch
        mbatch = MaskedBatch(src=src, tgt=tgt, pad=0)

        # loss
        with self.accelerator.autocast():
            preds = net(mbatch.src, mbatch.tgt, mbatch.src_mask, mbatch.tgt_mask)
            preds = preds.reshape(-1, preds.size(-1))
            labels = mbatch.tgt_y.reshape(-1)
            loss = loss_fn(preds, labels) / mbatch.ntokens

            # filter padding
            preds = preds.argmax(dim=-1).view(-1)[labels != 0]
            labels = labels[labels != 0]

        # backward()
        if self.stage == "train" and self.optimizer is not None:
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        all_loss = self.accelerator.gather(loss).sum()
        all_preds = self.accelerator.gather(preds)
        all_labels = self.accelerator.gather(labels)

        # losses (or plain metrics that can be averaged)
        step_losses = {self.stage + "_loss": all_loss.item()}

        step_metrics = {self.stage + "_" + name: metric_fn(all_preds, all_labels).item()
                        for name, metric_fn in self.metrics_dict.items()}

        if self.stage == "train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses, step_metrics


KerasModel.StepRunner = StepRunner

from torchmetrics import Accuracy

net = Transformer.from_config(src_vocab=len(vocab_x), tgt_vocab=len(vocab_y), N=5, d_model=64, d_ff=128, h=8, dropout=0.1)

loss_fn = LabelSmoothingLoss(size=len(vocab_y), padding_idx=0, smoothing=0.1)

metrics_dict = {'acc': Accuracy(task='multiclass', num_classes=len(vocab_y))}
optimizer = NoamOpt(net.parameters(), model_size=64)

model = KerasModel(net,
                   loss_fn=loss_fn,
                   metrics_dict=metrics_dict,
                   optimizer=optimizer)

model.fit(
    train_data=dl_train,
    val_data=dl_val,
    epochs=100,
    ckpt_path='checkpoint.pth',
    patience=10,
    monitor='val_acc',
    mode='max',
    callbacks=None,
    plot=True
)
