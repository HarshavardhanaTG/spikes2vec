import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from torch.utils.data import DataLoader
from spikes.encoder import computeMaskIndices

def makeData(fileName):
    path = '/mnt/data/HarshaData/StanfordT12/competitionData/Codes/'
    with open(path + fileName, 'rb') as file:
        trainData = pickle.load(file)

    dataChain = []
    for i in range(trainData["cue"].shape[0]):
        dataChain.append(trainData["spikePower"][i, :128, :trainData["actualLength"][i]])
    dataChain = np.concatenate(dataChain, axis = 1)

    timeLength = 150
    channels = 128
    numberSlices = dataChain.shape[1] // timeLength

    dataSlices = np.zeros((numberSlices, channels, timeLength))
    for i in range(numberSlices):
        dataSlices[i] = dataChain[:, i * timeLength: i * timeLength + timeLength]

    return dataSlices

class DataCollator:  

    def __init__(self):
        self.timeLength = 150
        self.channels = 128

    def __call__(self, features):
        featuresStacked = torch.stack(features, dim = 0)
        batchSize = featuresStacked.shape[0]

        maskTimeIndices = computeMaskIndices(
            shape = (batchSize, self.timeLength),
            maskProb = 0.65,
            maskLength = 10,
            paddingMask = None,
        )

        return featuresStacked, maskTimeIndices