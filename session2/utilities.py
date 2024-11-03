import math 
class LrUtilities():
    class WarmupScheduler:

        def __init__(self, optimizer, steps):
            self.optimizer = optimizer
            self.steps = steps
            self.tgt_lr = self.optimizer.param_groups[0]['lr']

        def step(self,epoch):
            if epoch <= self.steps:
                lr = (epoch / self.steps) * self.tgt_lr
            else:
                lr = self.tgt_lr

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    class CosineAnnealing :

        def __init__(self,optimizer,eta_min = 0) :
            self.optimizer = optimizer
            self.eta_max = self.optimizer.param_groups[0]["lr"]
            self.eta_min = eta_min

        def step(self,T_curr,T_max):
            lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) *(1+ math.cos(math.pi * T_curr/ T_max))      
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr