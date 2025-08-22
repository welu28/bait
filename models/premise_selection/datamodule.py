import pickle
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from data.utils.graph_data_utils import transform_expr, transform_batch


class PremiseDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage: str) -> None:
        source = self.config.source

        if source == 'mongodb':
            raise NotImplementedError(
                "MongoDB source disabled. Use 'directory' with a pickle file instead."
            )

        elif source == 'directory':
            data_dir = self.config.data_options['directory']  # path to your .pk file
            with open(data_dir, 'rb') as f:
                self.data = pickle.load(f)

            # load vocab + expression graphs
            self.vocab = self.data['vocab']
            self.expr_dict = {k: self.to_data(v) for k, v in self.data['expr_dict'].items()}

            # splits
            self.train_data = self.data['train_data']
            self.val_data = self.data['val_data']
            self.test_data = self.data['test_data']

        else:
            raise NotImplementedError

    def transfer_batch_to_device(self, batch, device: torch.device, dataloader_idx: int):
        if self.config.type == 'custom':
            pass
        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch

    def list_to_data(self, data_list):
        # always use in-memory dictionary, since we no longer stream from Mongo
        batch = [self.expr_dict[d] for d in data_list]
        return transform_batch(batch, config=self.config)

    def to_data(self, expr):
        return transform_expr(expr, self.config.type, self.vocab, self.config)

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