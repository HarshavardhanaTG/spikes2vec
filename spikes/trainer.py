import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from omegaconf import DictConfig
from tqdm import tqdm

from spikes.dataMaker import makeData, DataCollator
from spikes.encoder import neuralModel
from data2vec import Data2Vec
from utils import AverageMeter, maybeSaveCheckpoint

class spikeTrainer(object):
    def __init__(self):
        super(spikeTrainer, self).__init__()
    
        self.numEpochs = 1000
        self.device = 'cuda'
        self.ckptDir = 'spikes/checkpoints/spikes2vecPretraining'
        self.saveCkptFreq = 10
        
        self.encoder = neuralModel()
        self.model = Data2Vec(encoder = self.encoder)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), 0.0005)
        self.criterion = nn.MSELoss(reduction = 'none')
        self.criterion.to(self.device)
        
        self.trainDataset = torch.tensor(makeData('trainDataNorm.pkl'), dtype = torch.float32)
        self.valDataset = torch.tensor(makeData('valDataNorm.pkl'), dtype = torch.float32)
        self.dataCollator = DataCollator()
        self.trainLoader = DataLoader(self.trainDataset, batch_size = 16, collateFn = self.dataCollator)
        self.valLoader = DataLoader(self.valDataset, batch_size = 16, collate_fn = self.dataCollator)

        self.tensorboard = SummaryWriter(log_dir = 'spikes/logs')
        self.lossTracker = AverageMeter('loss')
        

    def trainStep(self, batch):
        
        src, mask = batch
        src, mask = src.to(self.device), mask.to(self.device)
        x, y = self.model(src, src, mask)
        loss = self.criterion(x.float(), y.float()).sum(dim=-1).div(x.size(0))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def testStep(self, batch):
        
        src, mask = batch
        src, mask = src.to(self.device), mask.to(self.device)
        x, y = self.model(src, src, mask = mask)
        loss = self.criterion(x.float(), y.float()).sum(dim=-1).div(x.size(0))

        return loss.item()

    def trainEpoch(self, epochNum):
        
        self.model.train()
        self.lossLracker.reset()
        with tqdm(self.trainLoader, unit = "batch", desc = f'Epoch: {epochNum}/{self.numEpochs} ',
                  bar_format = '{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii = " #") as iterator:
            for batch in iterator:
                loss = self.trainStep(batch)
                self.model.emaStep()
                self.lossTracker.update(loss)
                avgLoss = self.lossTracker.avg
                iterator.set_postfix(loss = avgLoss)

        return avgLoss

    def evaluate(self):
        
        self.model.eval()
        self.lossTracker.reset()
        with tqdm(self.valLoader, unit = "batch", desc = f'Evaluating... ',
                  bar_format = '{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii = " #") as iterator:
            with torch.no_grad():
                for batch in iterator:
                    loss = self.testStep(batch)
                    self.lossTracker.update(loss)
                    avgLoss = self.lossTracker.avg
                    iterator.set_postfix(loss = avgLoss)

        return avgLoss

    def train(self):
        
        for epoch in range(1, self.numEpochs + 1):
            print()
            trainLoss = self.trainEpoch(epoch)
            valLoss = self.evaluate()

            self.tensorboard.add_scalar('trainLoss', trainLoss, epoch)
            self.tensorboard.add_scalar('valLoss', valLoss, epoch)

            maybeSaveCheckpoint(self.model, self.optimizer, self.ckptDir, epoch, self.saveCkptFreq)