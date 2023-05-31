import os
from pathlib import Path


import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch import loggers
import torch

from src.model import BPModel
from src.dataset import BPDeepRNNDataset, load_from_npy, load_test_npy
from src.deeprnn import DeepRNN
from src.utils import load_configs


class BPDeepRNNTrainer:
    def __init__(self, config_filepath="config_deeprnn.yml"):
        self.config = load_configs(config_filepath)

        self.output_dir = self.config["output_dir"]
        self.data_dir = self.config["data_dir"]
        self.batch_size = self.config["batch_size"]
        self.deeprnn_config = self.config["deeprnn"]
        self.trainer_config = self.config["trainer"]

        os.makedirs(self.output_dir, exist_ok=True)

        self.model = None
        self.trainer = self.get_trainer()

    def get_dataloader(self):
        data = load_from_npy(self.data_dir, "deeprnn", deeprnn=True)

        train_data = BPDeepRNNDataset(
            data["X_train"], data["X_train_masks"], data["y_train"]
        )
        eval_data = BPDeepRNNDataset(
            data["X_test"], data["X_test_masks"], data["y_test"]
        )

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, num_workers=4
        )

        return {"train_dataloaders": train_loader, "val_dataloaders": eval_loader}

    def get_model(self):
        model = DeepRNN(**self.deeprnn_config)
        return BPModel(model)

    def get_trainer(self):
        callbacks = self.get_callbacks()
        self.trainer_config["callbacks"] = callbacks

        tb_logger = loggers.TensorBoardLogger(save_dir=self.output_dir)
        return pl.Trainer(logger=tb_logger, **self.trainer_config)

    def get_callbacks(self):
        return [
            EarlyStopping(monitor="Val Loss", min_delta=1.0, patience=3, mode="min")
        ]

    def train(self):
        training_components = {}

        loaders = self.get_dataloader()
        training_components.update(loaders)

        self.model = self.get_model()
        training_components["model"] = self.model

        self.trainer.fit(**training_components)

    def get_test_set(self):
        x_test, x_test_masks, y_test = load_test_npy(self.data_dir, deeprnn=True)
        dataset = BPDeepRNNDataset(x_test, x_test_masks, y_test)

        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, num_workers=8
        )
        return test_loader

    def test(self):
        loaders = self.get_test_set()
        results = self.trainer.test(dataloaders=loaders)

        results_file = Path(self.output_dir, "results.txt")
        with open(results_file, "w") as f:
            f.write(f"MSE on Test set: {results[0]['Test Loss']} \n")


if __name__ == "__main__":
    bptrainer = BPDeepRNNTrainer()
    bptrainer.train()
    bptrainer.test()
