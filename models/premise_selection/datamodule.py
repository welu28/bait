import pickle
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from data.utils.graph_data_utils import transform_expr, transform_batch


class PremiseDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage: str = None) -> None:
        # Use directory with pickle files
        data_dir = self.config.data_options['directory']  # path to your pickle folder

        # Load pickles
        with open(f"{data_dir}/expr_dict.pkl", "rb") as f:
            expr_dict = pickle.load(f)
        with open(f"{data_dir}/train.pkl", "rb") as f:
            train_pairs = pickle.load(f)
        with open(f"{data_dir}/val.pkl", "rb") as f:
            val_pairs = pickle.load(f)
        with open(f"{data_dir}/test.pkl", "rb") as f:
            test_pairs = pickle.load(f)
        with open(f"{data_dir}/vocab.pkl", "rb") as f:
            vocab = pickle.load(f)

        # Transform expressions
        self.vocab = vocab
        self.expr_dict = {k: self.to_data(v) for k, v in expr_dict.items()}

        # Splits
        self.train_data = train_pairs
        self.val_data = val_pairs
        self.test_data = test_pairs

    def transfer_batch_to_device(self, batch, device: torch.device, dataloader_idx: int):
        if getattr(self.config, 'type', None) == 'custom':
            pass
        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch

    def list_to_data(self, data_list):
        # always use in-memory dictionary
        batch = [self.expr_dict[d] for d in data_list]
        return transform_batch(batch, config=self.config)

    def to_data(self, expr):
        return transform_expr(expr, getattr(self.config, 'type', None), self.vocab, self.config)

    def collate_data(self, batch):
        y = torch.LongTensor([b['y'] for b in batch])
        data_1 = self.list_to_data([b['conj'] for b in batch])
        data_2 = self.list_to_data([b['stmt'] for b in batch])
        return data_1, data_2, y

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_data,
            shuffle=self.config.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_data,
            shuffle=self.config.shuffle,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_data,
        )