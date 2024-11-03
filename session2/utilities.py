import torch.optim as optim
class LrUtilities():
    class WarmupScheduler:
        def __init__(self, optimizer, steps):
            self.optimizer = optimizer
            self.steps = steps
            self.tgt_lr = optimizer.param_groups[0]['lr']

        def step(self,epoch):
            if epoch <= self.steps:
                lr = (epoch / self.steps) * self.tgt_lr
            else:
                lr = self.tgt_lr

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    class rate_scheduler:
        def __init__(self,optimizer) -> None:
            self.optimizer = optimizer
            pass