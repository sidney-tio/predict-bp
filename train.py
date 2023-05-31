import torch
import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch import loggers
from pathlib import Path

from src.model import BPModel
from src.dataset import BPDataset, load_from_npy, load_test_npy
from src.resnet import ResNet1D
from src.utils import load_configs


class BPTrainer():
    def __init__(self, config_filepath = 'config.yml'):
        self.config = load_configs(config_filepath)

        self.output_dir = self.config['output_dir']
        self.data_dir = self.config['data_dir']
        self.batch_size = self.config['batch_size']
        self.resnet_config = self.config['resnet']
        self.trainer_config = self.config['trainer']

        os.makedirs(self.output_dir, exist_ok = True)

        self.model = None
        self.trainer = self.get_trainer()

    def get_dataloader(self):
        data = load_from_npy(self.data_dir, 'vitals')

        train_data = BPDataset(data['X_train'], data['y_train'])
        eval_data =  BPDataset(data['X_test'], data['y_test'])

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size = self.batch_size,
                                                   shuffle = True,
                                                   num_workers = 12)
        eval_loader = torch.utils.data.DataLoader(eval_data,
                                                  batch_size = self.batch_size,
                                                  num_workers = 8
                                                )

        return {
            "train_dataloaders": train_loader,
            "val_dataloaders": eval_loader
        }

    def get_model(self):
        model = ResNet1D(**self.resnet_config)
        model.double()
        return BPModel(model)

    def get_trainer(self):
        callbacks = self.get_callbacks()
        self.trainer_config['callbacks'] = callbacks

        tb_logger = loggers.TensorBoardLogger(save_dir = self.output_dir)
        return pl.Trainer(logger = tb_logger,
                          **self.trainer_config)

    def get_callbacks(self):
        return [EarlyStopping(monitor = 'Val Loss',
                              min_delta = 1.0,
                              patience = 3,
                              mode = 'min')]

    def train(self):
        training_components = {}

        loaders = self.get_dataloader()
        training_components.update(loaders)

        self.model = self.get_model()
        training_components['model'] = self.model

        self.trainer.fit(**training_components)

    def get_test_set(self):
        x_test, y_test = load_test_npy(self.data_dir)
        dataset = BPDataset(x_test, y_test)

        test_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size = self.batch_size,
                                                  num_workers = 8
                                                )
        return test_loader

    def test(self):
        loaders = self.get_test_set()
        results = self.trainer.test(dataloaders = loaders)

        results_file = Path(self.output_dir, "results.txt")
        with open(results_file, 'w') as f:
            f.write(f"MSE on Test set: {results[0]['Test Loss']} \n")


if __name__ == "__main__":
    bptrainer = BPTrainer()
    bptrainer.train()
    bptrainer.test()