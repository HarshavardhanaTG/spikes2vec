import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
from torch import Tensor
from typing import List, Optional, Tuple, Union

def computeMaskIndices(
    shape: Tuple[int, int],
    paddingMask: Optional[Tensor],
    maskProb: float,
    maskLength: int,
    maskType: str = "static",
    maskOther: float = 0.0,
    minMasks: int = 0,
    noOverlap: bool = False,
    minSpace: int = 0,
) -> Tensor:
    
    batchSize, frame = shape
    mask = torch.full((batchSize, frame), False)
    
    allNumMask = int(maskProb * frame / float(maskLength) + torch.rand(1))

    allNumMask = max(minMasks, allNumMask)

    maskIdcs = []
    for i in range(batchSize):
        if paddingMask is not None:
            sz = frame - paddingMask[i].long().sum().item()
            print(sz)
            numMask = int(maskProb * sz / float(maskLength) + torch.rand(1))
            numMask = max(minMasks, numMask)
        else:
            sz = frame
            numMask = allNumMask

        if maskType == "static":
            lengths = torch.full((numMask,), maskLength)
        elif maskType == "uniform":
            lengths = torch.randint(int(maskOther), maskLength * 2 + 1, size = (numMask,))
        elif maskType == "normal":
            lengths = torch.normal(maskLength, maskOther, size = (numMask,))
            lengths = torch.maximum(torch.ones(1), torch.round(lengths)).int()
        elif maskType == "poisson":
            lengths = torch.poisson(maskLength, size = (numMask,))
            lengths = torch.round(lengths).int()
        else:
            raise Exception(f"unknown mask selection: {maskType}")

        if sum(lengths) == 0:
            lengths[0] = min(maskLength, sz - 1)

        if noOverlap:
            maskIdc = []

            def arrange(s, e, length, keepLength):
                spanStart = torch.randint(s, e - length, size = (1,))
                maskIdc.extend(spanStart + i for i in range(length))

                newParts = []
                if spanStart - s - minSpace >= keepLength:
                    newParts.append((s, spanStart - minSpace + 1))
                if e - spanStart - keepLength - minSpace > keepLength:
                    newParts.append((spanStart + length + minSpace, e))
                return newParts

            parts = [(0, sz)]
            minLength = min(lengths)
            for length in sorted(lengths, reverse = True):
                lens = torch.tensor([e - s for s, e in parts], dtype = torch.int)
                lens[lens < length + minSpace] = 0
                lSum = lens.sum()
                if lSum == 0:
                    break
                probs = lens / lSum
                c = torch.distributions.categorical.Categorical(probs).sample()
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, minLength))
            maskIdc = torch.tensor(maskIdc)
        else:
            minLen = min(lengths)
            if sz - minLen <= numMask:
                minLen = sz - numMask - 1

            maskIdc = torch.randperm(sz - minLen)[:numMask]
            maskIdc = torch.tensor(
                [maskIdc[j] + offset for j in range(len(maskIdc)) for offset in range(lengths[j])]
            )

        maskIdcs.append(torch.unique(maskIdc[maskIdc < sz]))

    minLen = min([len(m) for m in maskIdcs])
    for i, maskIdc in enumerate(maskIdcs):
        if len(maskIdc) > minLen:
            maskIdc = maskIdc[torch.randperm(len(maskIdc))[:minLen].long()]
        mask[i, maskIdc] = True

    return mask

class neuralResidualBlock(nn.Module):
    def __init__(self):
        super(neuralResidualBlock, self).__init__()
        
        self.conv_11 = nn.Conv1d(128, 256, kernel_size = 3, stride = 1, padding = 'same', dilation = 4)
        self.conv_12 = nn.Conv1d(256, 256, kernel_size = 3, stride = 1, padding = 'same', dilation = 8)
        self.conv_13 = nn.Conv1d(256, 512, kernel_size = 3, stride = 1, padding = 'same', dilation = 2)

        self.conv_21 = nn.Conv1d(256, 256, kernel_size = 3, stride = 1, padding = 'same', dilation = 16)
        self.conv_22 = nn.Conv1d(256, 256, kernel_size = 3, stride = 1, padding = 'same', dilation = 1)
        self.conv_23 = nn.Conv1d(256, 512, kernel_size = 3, stride = 1, padding = 'same', dilation = 2)

        self.conv_31 = nn.Conv1d(256, 256, kernel_size = 3, stride = 1, padding = 'same', dilation = 2)
        self.conv_32 = nn.Conv1d(256, 256, kernel_size = 3, stride = 1, padding = 'same', dilation = 4)
        self.conv_33 = nn.Conv1d(256, 512, kernel_size = 3, stride = 1, padding = 'same', dilation = 2)

        self.conv_41 = nn.Conv1d(256, 256, kernel_size = 3, stride = 1, padding = 'same', dilation = 8)
        self.conv_42 = nn.Conv1d(256, 256, kernel_size = 3, stride = 1, padding = 'same', dilation = 16)
        self.conv_43 = nn.Conv1d(256, 512, kernel_size = 3, stride = 1, padding = 'same', dilation = 2)

        self.conv_51 = nn.Conv1d(256, 256, kernel_size = 3, stride = 1, padding = 'same', dilation = 1)
        self.conv_52 = nn.Conv1d(256, 256, kernel_size = 3, stride = 1, padding = 'same', dilation = 2)
        self.conv_53 = nn.Conv1d(256, 512, kernel_size = 3, stride = 1, padding = 'same', dilation = 2)

        self.batchNorm = nn.BatchNorm1d(256)
        self.activation = nn.GELU()
        self.glu = nn.GLU(dim = 1)

    def forward(self, x):

        x = self.activation(self.batchNorm(self.conv_11(x)))
        residual = x
        x = self.activation(self.batchNorm(self.conv_12(x)))
        x += residual
        x = self.glu(self.conv_13(x))

        residual = x
        x = self.activation(self.batchNorm(self.conv_21(x)))
        x += residual
        residual = x
        x = self.activation(self.batchNorm(self.conv_22(x)))
        x += residual
        x = self.glu(self.conv_23(x))

        residual = x
        x = self.activation(self.batchNorm(self.conv_31(x)))
        x += residual
        residual = x
        x = self.activation(self.batchNorm(self.conv_32(x)))
        x += residual
        x = self.glu(self.conv_33(x))

        residual = x
        x = self.activation(self.batchNorm(self.conv_41(x)))
        x += residual
        residual = x
        x = self.activation(self.batchNorm(self.conv_42(x)))
        x += residual
        x = self.glu(self.conv_43(x))

        residual = x
        x = self.activation(self.batchNorm(self.conv_51(x)))
        x += residual
        residual = x
        x = self.activation(self.batchNorm(self.conv_52(x)))
        x += residual
        x = self.glu(self.conv_53(x))

        return x
    
class neuralProjectionModel(nn.Module):
    def __init__(self):
        super(neuralProjectionModel, self).__init__()

        self.convBlocks = neuralResidualBlock()
        self.finalConv1 = nn.Conv1d(256, 512, kernel_size = 1)
        self.finalGelu = nn.GELU()

        self.norm = nn.LayerNorm(512)
        self.finalProjection = nn.Linear(512, 768)
        self.dropout = nn.Dropout(0.1)
        

    def forward(self, x):
        x = self.convBlocks(x)

        x = self.finalConv1(x)
        x = self.finalGelu(x)

        x = self.norm(x.transpose(1, 2))
        x = self.finalProjection(x)
        x = self.dropout(x)

        return x
    
class ConvolutionalPositionalEmbedding(nn.Module):

    def __init__(self):
        super(ConvolutionalPositionalEmbedding, self).__init__()
        
        self.conv = nn.Conv1d(in_channels = 768, out_channels = 768, kernel_size = 128, padding = 64, groups = 16)

        self.conv = nn.utils.parametrizations.weight_norm(self.conv, name = "weight", dim = 2)
        self.numRemove: int = 1 if 128 % 2 == 0 else 0

    def forward(self, x):
        x = x.transpose(-2, -1)
        x = self.conv(x)
        if self.numRemove > 0:
            x = x[..., :-self.numRemove]
        x = torch.nn.functional.gelu(x)
        x = x.transpose(-2, -1)
        return x
    
class Wav2Vec2FeedForward(nn.Module):
    def __init__(self):
        super(Wav2Vec2FeedForward, self).__init__()
        self.intermediateDropout = nn.Dropout(0.1)
        self.intermediateDense = nn.Linear(768, 3072)
        self.outputDense = nn.Linear(3072, 768)
        self.outputDropout = nn.Dropout(0.1)

    def forward(self, hiddenStates):
        hiddenStates = self.intermediateDense(hiddenStates)
        hiddenStates = torch.nn.functional.gelu(hiddenStates)
        hiddenStates = self.intermediateDropout(hiddenStates)

        hiddenStates = self.outputDense(hiddenStates)
        hiddenStates = self.outputDropout(hiddenStates)
        return hiddenStates
    
class Wav2Vec2Attention(nn.Module):
    
    def __init__(self):
        super(Wav2Vec2Attention, self).__init__()
        headDim = 768 // 12
        
        self.embedDim = 768
        self.numHeads = 12
        self.dropout = 0.1
        self.headDim = headDim

        self.scaling = self.headDim ** -0.5

        self.kProj = nn.Linear(768, 768, bias = True)
        self.vProj = nn.Linear(768, 768, bias = True)
        self.qProj = nn.Linear(768, 768, bias = True)
        self.outProj = nn.Linear(768, 768, bias = True)

    def forward(self, x, attentionMask: Optional[Tensor] = None)-> Tuple[Tensor, Optional[Tensor]]:
        
        batchSize, length, embedDim = x.size()
        
        shape = (batchSize, length, self.numHeads, self.headDim)
        q = self.qProj(x).view(*shape).transpose(2, 1)
        k = self.kProj(x).view(*shape).transpose(2, 1)  
        v = self.vProj(x).view(*shape).transpose(2, 1)  
        dropout = self.dropout if self.training else 0.0
        attnOutput = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = attentionMask, dropout_p = dropout, is_causal = False)
        attnOutput = attnOutput.transpose(1, 2).reshape(batchSize, -1, self.numHeads * self.headDim)
        output = self.outProj(attnOutput)
        return output, None  
    
class Wav2Vec2EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Wav2Vec2Attention()
        self.dropout = nn.Dropout(0.1)
        self.layerNorm = nn.LayerNorm(768)
        self.feedForward = Wav2Vec2FeedForward()
        self.finalLayerNorm = nn.LayerNorm(768)

    def forward(self, hiddenStates, attentionMask = None):
        attnResidual = hiddenStates
        hiddenStates, _ = self.attention(hiddenStates, attentionMask = attentionMask)
        hiddenStates = self.dropout(hiddenStates)
        hiddenStates = attnResidual + hiddenStates
        hiddenStates = self.layerNorm(hiddenStates)
        hiddenStates = hiddenStates + self.feedForward(hiddenStates)
        hiddenStates = self.finalLayerNorm(hiddenStates)

        outputs = (hiddenStates,)
        return outputs
    
class neuralEncoder(nn.Module):
    def __init__(self):
        super(neuralEncoder, self).__init__()
        self.posConvEmbed = ConvolutionalPositionalEmbedding()
        self.layerNorm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([Wav2Vec2EncoderLayer() for _ in range(12)])
    
    def forward(
        self,
        hiddenStates: torch.tensor,
        attentionMask: Optional[torch.Tensor] = None,
        outputAttentions: bool = False,
        outputHiddenStates: bool = False,
    ):
        
        positionEmbeddings = self.posConvEmbed(hiddenStates)
        hiddenStates = hiddenStates + positionEmbeddings
        hiddenStates = self.layerNorm(hiddenStates)
        hiddenStates = self.dropout(hiddenStates)

        ret: List[Tensor] = []
        for layer in self.layers:
            if not (self.training and torch.rand(1).item() <= 0.1):
                layerOutputs = layer(hiddenStates, attentionMask)
                hiddenStates = layerOutputs[0]
            ret.append(hiddenStates)
            
        return hiddenStates, ret
    
class neuralModel(nn.Module):
    def __init__(self):
        super(neuralModel, self).__init__()
    
        self.featureExtractor = neuralProjectionModel()
        self.maskedSpecEmbed = nn.Parameter(torch.FloatTensor(768).uniform_())
        self.encoder = neuralEncoder()


    def _maskHiddenStates(self, hiddenStates: torch.FloatTensor, maskTimeIndices: Optional[torch.FloatTensor] = None, attentionMask: Optional[torch.LongTensor] = None,):
        
        batchSize, sequenceLength, hiddenSize = hiddenStates.size()

        if maskTimeIndices is not None:
            hiddenStates[maskTimeIndices] = self.maskedSpecEmbed.to(hiddenStates.dtype)
        
        return hiddenStates

    def forward(
        self,
        inputValues: Optional[torch.Tensor],
        attentionMask: Optional[torch.Tensor] = None,
        maskTimeIndices: Optional[torch.FloatTensor] = None,
        outputAttentions: Optional[bool] = None,
        outputHiddenStates: Optional[bool] = None,
        returnDict: Optional[bool] = None,):

        hiddenStates = self.featureExtractor(inputValues)
        hiddenStates = self._maskHiddenStates(hiddenStates, maskTimeIndices = maskTimeIndices, attentionMask = attentionMask)

        lastLayer, allLayers = self.encoder(
            hiddenStates,
            attentionMask = attentionMask,
            outputAttentions = outputAttentions,
            outputHiddenStates = outputHiddenStates,
        )

        
        return lastLayer, allLayers
    
