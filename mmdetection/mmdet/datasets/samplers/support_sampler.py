# Copyright (c) OpenMMLab. All rights reserved.
from typing import Iterator, Optional, Sized

from torch.utils.data import Sampler

from mmdet.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class SupportSampler(Sampler):
    """Iterate over the complete support set in deterministic order.

    Unlike distributed samplers, every rank visits the full support set. This
    keeps the cached support tokens identical across test workers.
    """

    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = False,
                 seed: Optional[int] = None) -> None:
        if shuffle:
            raise ValueError('SupportSampler does not support shuffling.')
        self.dataset = dataset

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.dataset)))

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        pass
