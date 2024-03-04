import omegaconf
from spikes.trainer import spikeTrainer

if __name__ == '__main__':
    import argparse
    trainer = spikeTrainer()
    trainer.train()