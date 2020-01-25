import torch.utils.data as data

from spoint.datasets import get_dataset


class Loader(data.DataLoader):
    def __init__(self, config):
        super().__init__(get_dataset(config),
                         batch_size=config.batch_size,
                         num_workers=config.num_workers)

        self.config = config

    @property
    def epochs(self):
        return self.config.epochs

    @property
    def start_epoch(self):
        return self.config.start_epoch
