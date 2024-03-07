import torch
import torch.nn as nn
import torch.nn.functional as F
from .ema import EMA


class Data2Vec(nn.Module):

    def __init__(self, encoder):
        super(Data2Vec, self).__init__()
        
        self.embedDim = 768
        self.encoder = encoder

        self.ema = EMA(self.encoder)
        self.regressionHead = self._buildRegressionHead()

        self.emaDecay = 0.9990
        self.emaEndDecay = 0.9999
        self.emaAnnealEndStep = 30000

    def _buildRegressionHead(self):
        
        return nn.Linear(self.embedDim, self.embedDim)

    def emaStep(self):
        
        if self.emaDecay != self.emaEndDecay:
            if self.ema.numUpdates >= self.emaAnnealEndStep:
                decay = self.emaEndDecay
            else:
                decay = self.ema.getAnnealedRate(
                    self.emaDecay,
                    self.emaEndDecay,
                    self.ema.numUpdates,
                    self.emaAnnealEndStep,
                )
            self.ema.decay = decay
        if self.ema.decay < 1:
            self.ema.step(self.encoder)

    def forward(self, src, trg = None, mask = None):
        
        xLastLayer, xAllLayers = self.encoder(inputValues = src, maskTimeIndices = mask)
        if trg is None:
            return xLastLayer

        with torch.no_grad():
            self.ema.model.eval()
            yLastLayer, yAllLayers = self.ema.model(inputValues = trg)
            yAllLayers = yAllLayers[-8:]
            
            
            yAllLayers = [F.instance_norm(tl.float()) for tl in yAllLayers]
            
            yAllLayers = sum(yAllLayers) / len(yAllLayers)
            yAllLayers = yAllLayers[mask]

        
        xLastLayer = xLastLayer[mask]
        
        xLastLayer = self.regressionHead(xLastLayer)

        return xLastLayer, yAllLayers