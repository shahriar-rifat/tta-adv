import torch

class AverageMeter():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0 
        self._avg = 0
        self._sum = 0
        self._count = 0

    def update(self, val, n=1):
        self.val = val
        self._sum += val*n
        self._count += n 
        self._avg = self._sum / self._count 
    
    @property
    def avg(self):
        return self._avg

def accuracy(output, target):
    acc = 0
    acc += (output.max(1)[1] == target).float().sum()
    batch_size = target.size()[0]
    return acc.mul_(100.0/batch_size)