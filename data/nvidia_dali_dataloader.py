from nvidia.dali.plugin.pytorch import to_torch_type, feed_ndarray
import nvidia.dali as dali
from distutils.version import LooseVersion

import torch
from torch.utils.data import DataLoader

import collections.abc as container_abcs
from torch._six import int_classes

import numpy as np


def _dataset_to_np_collate_fn(batch):
    elem_type = type(batch[0])
    if isinstance(batch[0], float):
        return [np.array([d], dtype=np.float64) for d in batch]
    elif isinstance(batch[0], int_classes):
        return [np.array([d], dtype=np.int32) for d in batch]
    elif elem_type.__module__ == 'numpy' and \
            elem_type.__name__ != 'str_' and \
            elem_type.__name__ != 'string_':
        return batch
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [_dataset_to_np_collate_fn(list(samples)) for samples in transposed]
    error_msg_fmt = "batch must contain tensors or lists; found {}"
    raise TypeError(error_msg_fmt.format(elem_type))


def _dali_run_one_step(pipeline, output_categories, data_batches, output_map,
                       current_data_batch):
    p = pipeline
    if LooseVersion(dali.__version__) <= LooseVersion('0.7.0'):
        p._prefetch()
    # p._prefetch()
    outputs = p._share_outputs()

    dev_id = p.device_id
    category_outputs = dict()

    for j, out in enumerate(outputs):
        category_outputs[output_map[j]] = out

    category_tensors = dict()
    category_shapes = dict()

    for category, out in category_outputs.items():
        category_tensors[category] = out.as_tensor()
        category_shapes[category] = category_tensors[category].shape()

    # if we did not yet allocate memory for that batch, do it now
    if data_batches[current_data_batch] is None:
        category_torch_type = dict()
        category_device = dict()
        torch_gpu_device = torch.device('cuda', dev_id)
        torch_cpu_device = torch.device('cpu')

        for category in output_categories:
            category_torch_type[category] = to_torch_type[np.dtype(
                category_tensors[category].dtype())]
            from nvidia.dali.backend import TensorGPU
            if type(category_tensors[category]) is TensorGPU:
                category_device[category] = torch_gpu_device
            else:
                category_device[category] = torch_cpu_device

        pyt_tensors = dict()
        for category in output_categories:
            pyt_tensors[category] = torch.zeros(category_shapes[category],
                                                dtype=category_torch_type[category],
                                                device=category_device[category])
        data_batches[current_data_batch] = pyt_tensors
    else:
        pyt_tensors = data_batches[current_data_batch]

    for category, tensor in category_tensors.items():
        feed_ndarray(tensor, pyt_tensors[category])

    p._release_outputs()
    # p._start_run()
    if LooseVersion(dali.__version__) > LooseVersion('0.7.0'):
        p._run()
    else:
        p._start_run()

    return [data_batches[current_data_batch][key] for key in output_map]


dali_default_collate = _dataset_to_np_collate_fn


class DaliDataLoader(DataLoader):
    r"""
    Pytorch Dataloder for dali.

    provide pytorch dataloader or dataset

    Arguments:
        pipeline (Pipeline):
        dataloader (DataLoader): None
        dataset (Dataset): None
        batch_size (int, optional): 1
        shuffle (bool optional): False
        sampler (Sampler): None
        batch_sampler (Sampler): None
        collate_fn (callable, optional): default_fn
        drop_last (bool optional): True
    """
    __initialized = False

    def __init__(self, pipeline, batch_size=1, dataloader=None,
                 dataset=None, sampler=None, batch_sampler=None,
                 num_workers=0, shuffle=False, timeout=0,
                 collate_fn=dali_default_collate, drop_last=False, verbose=False):

        self.pipeline = pipeline
        self.dataloader = dataloader
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn

        self.verbose = verbose

        if (self.dataloader is not None) and (dataset is not None):
            raise ValueError('dataloader and dataset is mutualy exclusive')

        if self.dataloader is None:
            self.dataloader = DataLoader(self.dataset,
                                         batch_size=self.batch_size,
                                         shuffle=shuffle,
                                         num_workers=num_workers,
                                         sampler=sampler,
                                         batch_sampler=batch_sampler,
                                         timeout=timeout,
                                         collate_fn=self.collate_fn,
                                         drop_last=drop_last,
                                         pin_memory=True)

        self.__initialized = True

    def __iter__(self):
        return _DaliDataLoaderIter(self)

    def __len__(self):
        return len(self.dataloader.batch_sampler)

    def __del__(self):
        del self.dataloader


class _DaliDataLoaderIter(object):

    def __init__(self, loader):

        self.pipeline = loader.pipeline
        self.pydataloader = loader.dataloader
        self.verbose = loader.verbose

        self.size = len(loader)

        self.pyt_iter = self.pyt_dataset_batch_iter()

        self.pipeline.build()

        self.pipeline.iter_setup = self.pipeline_iter_setup().__get__(
            self.pipeline)
        self.pipeline.iter = 0
        self.output_map = ['data', 'label']
        self._output_categories = set(self.output_map)
        self._data_batches = [None, None]
        self._counter = 0
        self._current_data_batch = 0
        self._last_batch_size = 0
        # self._first_batch = None
        self._data_read_time = 0

        if LooseVersion(dali.__version__) > LooseVersion('0.7.0'):
            self.pipeline._run()

        self._prefetch_counter = 0
        self._first_batch = None

        self._first_batch = self.next()

    def __len__(self):
        return len(self.pydataloader)

    def pyt_dataset_batch_iter(self):
        for batch in self.pydataloader:
            yield batch

    def pipeline_iter_setup(self):
        owner = self
        np_data_iter = self.pyt_iter
        import time

        def iter_setup(self):
            s = time.time()
            raw_datas, raw_labels = next(np_data_iter)
            owner._data_read_time += time.time() - s
            assert(len(raw_datas) == len(raw_labels))
            # if this is the last batch, which may be less than batch size
            # dali can't handle such case, so we add dummy examples here
            # and delete later
            if len(raw_datas) < self._batch_size:
                owner._last_batch_size = len(raw_datas)
                extra_size = self._batch_size - owner._last_batch_size

                raw_datas += raw_datas[:1] * extra_size
                raw_labels += raw_labels[:1] * extra_size

            self.feed_input(self.datas, raw_datas)
            self.feed_input(self.labels, raw_labels)
            self.iter += 1
        return iter_setup

    def reset(self):
        self.pipeline.reset()

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch

        if self.pipeline._last_iter:
            self._prefetch_counter += 1
            if self._prefetch_counter == self.pipeline._prefetch_queue_depth:
                self.reset()
                raise StopIteration

        batch = _dali_run_one_step(self.pipeline, self._output_categories,
                                   self._data_batches, self.output_map,
                                   self._current_data_batch)

        self._current_data_batch = (self._current_data_batch + 1) % 2
        self._counter += 1

        if self.verbose and self._counter % 10 == 0:
            print('data load time(avg): {} s'.format(
                self._data_read_time / self._counter))

        # chop off last batch
        if self._counter >= self.size and self._last_batch_size > 0:
            if self.verbose:
                print('last batch: {} {}'.format(self._counter,
                                                 self._last_batch_size))

            batch = [item[:self._last_batch_size] for item in batch]

        return batch

    next = __next__

    def __iter__(self):
        return self
