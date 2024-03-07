import os
import copy

import torch
import torch.nn as nn


class EMA:

    def __init__(self, model: nn.Module, skipKeys = None):
        self.model = self.deepcopyModel(model)
        self.model.requires_grad_(False)
        self.device = 'cuda'
        self.model.to(self.device)
        self.skipKeys = skipKeys or set()
        self.decay = 0.9990
        self.numUpdates = 0

    @staticmethod
    def deepcopyModel(model):
        try:
            model = copy.deepcopy(model)
            return model
        except RuntimeError:
            tmpPath = 'tmpModelForEmaDeepcopy.pt'
            torch.save(model, tmpPath)
            model = torch.load(tmpPath)
            os.remove(tmpPath)
            return model

    def step(self, newModel: nn.Module):
        
        emaStateDict = {}
        emaParams = self.model.state_dict()
        for key, param in newModel.state_dict().items():
            emaParam = emaParams[key].float()
            if key in self.skipKeys:
                emaParam = param.to(dtype = emaParam.dtype).clone()
            else:
                emaParam.mul_(self.decay)
                emaParam.add_(param.to(dtype = emaParam.dtype), alpha = 1 - self.decay)
            emaStateDict[key] = emaParam
        self.model.load_state_dict(emaStateDict, strict = False)
        self.numUpdates += 1

    def restore(self, model: nn.Module):
        
        d = self.model.state_dict()
        model.load_state_dict(d, strict = False)
        return model

    def state_dict(self):
        return self.model.state_dict()

    @staticmethod
    def getAnnealedRate(start, end, currStep, totalSteps):
        
        r = end - start
        pctRemaining = 1 - currStep / totalSteps
        return end - r * pctRemaining