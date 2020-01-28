import abc
import threading
from multiprocessing import Process, SimpleQueue

import numpy as np

from spoint.datasets import get_dataset


class Sampler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass


class SequentialSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return iter(range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)


class RandomSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return iter(np.random.permutation(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)


class BatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []

        for idx in self.sampler:
            batch.append(idx)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class DataLoaderIter:
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.done_event = threading.Event()

        self.sample_iter = iter(self.batch_sampler)

        if self.num_workers > 0:
            self.index_queue = SimpleQueue()
            self.data_queue = SimpleQueue()
            self.batches_outstanding = 0
            self.shutdown = False
            self.send_idx = 0
            self.idx = 0
            self.reorder_dict = {}

            self.workers = [Process(target=self._loop, args=(self.dataset, self.index_queue, self.data_queue))
                            for _ in range(self.num_workers)]

            for w in self.workers:
                w.daemon = True
                w.start()

            for _ in range(2 * self.num_workers):
                self._put_indices()

    def _loop(self, dataset, index_queue, data_queue):
        while True:
            r = index_queue.get()

            if r is None:
                data_queue.put(None)
                break

            idx, batch_indices = r

            samples = [dataset[i] for i in batch_indices]
            data_queue.put((idx, samples))

    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):
        if self.num_workers == 0:
            indices = next(self.sample_iter)
            batch = [self.dataset[i] for i in indices]
            return batch

        if self.idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.idx)
            return self._process_next_batch(batch)

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)

            idx, batch = self.data_queue.get()
            self.batches_outstanding -= 1

            if idx != self.idx:
                self.reorder_dict[idx] = batch
                continue

            return self._process_next_batch(batch)

    def __iter__(self):
        return self

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers

        indices = next(self.sample_iter, None)

        if indices is None:
            return

        self.index_queue.put((self.send_idx, indices))
        self.batches_outstanding += 1
        self.send_idx += 1

    def _process_next_batch(self, batch):
        self.idx += 1
        self._put_indices()

        return batch

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True
            self.done_event.set()

            for _ in self.workers:
                self.index_queue.put(None)

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()


class DataLoader:
    def __init__(self, config):
        self.config = config

        batch_size = self.config.get('batch_size', 1)
        shuffle = self.config.get('shuffle', False)
        num_workers = self.config.get('num_workers', 1)
        drop_last = self.config.get('drop_last', False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last

        self.dataset = get_dataset(config)

        if shuffle:
            sampler = RandomSampler(self.dataset)
        else:
            sampler = SequentialSampler(self.dataset)

        self.sampler = sampler
        self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)

    def __iter__(self):
        return DataLoaderIter(self)

    def __len__(self):
        return len(self.batch_sampler)

    @property
    def epochs(self):
        return self.config.epochs

    @property
    def start_epoch(self):
        return self.config.start_epoch
