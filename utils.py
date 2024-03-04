import os
import torch


class AverageMeter(object):

    def __init__(self, name, fmt = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def maybeSaveCheckpoint(model, optimizer, path, epochNum, saveFreq):
    
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, f'{epochNum}.pt')
    if epochNum % saveFreq == 0:
        checkpoint = {'data2vec': model.state_dict(),
                      'encoder': model.encoder.encoder.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, path)
        print(f'Saved checkpoint to `{path}`')