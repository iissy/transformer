import torch

class NoamOpt(torch.optim.AdamW):
    def __init__(self, params, model_size=512, factor=1.0, warmup=4000,
                 lr=0, betas=(0.9, 0.98), eps=1e-9,
                 weight_decay=0, amsgrad=False):
        super(NoamOpt, self).__init__(params, lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay, amsgrad=amsgrad)
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size

    def step(self, closure=None):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.param_groups:
            p['lr'] = rate
        super(NoamOpt, self).step(closure=closure)

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step * self.warmup ** (-1.5), step ** (-0.5)))