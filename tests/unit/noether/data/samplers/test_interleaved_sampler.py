#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from noether.data.samplers import InterleavedSampler, InterleavedSamplerConfig, SamplerIntervalConfig


class ListDataset(Dataset):
    """Wraps a list into a dataset such that python types don't complain when passing lists to DistributedSampler."""

    def __init__(self, values: list[int]):
        super().__init__()
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.values[idx]


def _run(sampler, expected):
    actual = [i for (_, i) in sampler]
    assert expected == actual


def test_eval_epochs0():
    _run(
        sampler=InterleavedSampler(
            train_sampler=SequentialSampler(list(range(10))),
            callback_samplers=[
                SamplerIntervalConfig(
                    sampler=SequentialSampler(list(range(5))),
                    every_n_epochs=1,
                    pipeline=None,
                ),
            ],
            config=InterleavedSamplerConfig(
                batch_size=4,
                drop_last=False,
                max_epochs=0,
            ),
        ),
        expected=[
            # configs[0]
            10,
            11,
            12,
            13,
            14,
        ],
    )


def test_only_main_sampler():
    _run(
        sampler=InterleavedSampler(
            train_sampler=SequentialSampler(list(range(5))),
            config=InterleavedSamplerConfig(
                batch_size=4,
                drop_last=False,
                max_epochs=1,
            ),
        ),
        expected=[
            # configs[0]
            0,
            1,
            2,
            3,
            4,
        ],
    )


def test_eval_updates0():
    _run(
        sampler=InterleavedSampler(
            train_sampler=SequentialSampler(list(range(10))),
            callback_samplers=[
                SamplerIntervalConfig(
                    sampler=SequentialSampler(list(range(5))),
                    every_n_epochs=1,
                    pipeline=None,
                ),
            ],
            config=InterleavedSamplerConfig(
                batch_size=4,
                drop_last=False,
                max_updates=0,
            ),
        ),
        expected=[
            # configs[0]
            10,
            11,
            12,
            13,
            14,
        ],
    )


def test_eval_samples0():
    _run(
        sampler=InterleavedSampler(
            train_sampler=SequentialSampler(list(range(10))),
            callback_samplers=[
                SamplerIntervalConfig(
                    sampler=SequentialSampler(list(range(5))),
                    every_n_epochs=1,
                    pipeline=None,
                ),
            ],
            config=InterleavedSamplerConfig(
                batch_size=4,
                drop_last=False,
                max_samples=0,
            ),
        ),
        expected=[
            # configs[0]
            10,
            11,
            12,
            13,
            14,
        ],
    )


def test_sequential_nodroplast_ene1sequential():
    _run(
        sampler=InterleavedSampler(
            train_sampler=SequentialSampler(list(range(10))),
            callback_samplers=[
                SamplerIntervalConfig(sampler=SequentialSampler(list(range(5))), every_n_epochs=1, pipeline=None),
            ],
            config=InterleavedSamplerConfig(
                batch_size=4,
                drop_last=False,
                max_epochs=2,
            ),
        ),
        expected=[
            # main
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            # configs[0]
            10,
            11,
            12,
            13,
            14,
            # main
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            # configs[0]
            10,
            11,
            12,
            13,
            14,
        ],
    )


def test_sequential_nodroplast_ene1sequential_startepoch():
    _run(
        sampler=InterleavedSampler(
            train_sampler=SequentialSampler(list(range(10))),
            callback_samplers=[
                SamplerIntervalConfig(
                    sampler=SequentialSampler(list(range(5))),
                    every_n_epochs=1,
                    pipeline=None,
                ),
            ],
            config=InterleavedSamplerConfig(
                batch_size=4,
                start_epoch=1,
                drop_last=False,
                max_epochs=2,
            ),
        ),
        expected=[
            # main
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            # configs[0]
            10,
            11,
            12,
            13,
            14,
        ],
    )


def test_sequential_nodroplast_ene2sequential():
    _run(
        sampler=InterleavedSampler(
            train_sampler=SequentialSampler(list(range(10))),
            callback_samplers=[
                SamplerIntervalConfig(
                    sampler=SequentialSampler(list(range(5))),
                    every_n_epochs=2,
                    pipeline=None,
                ),
            ],
            config=InterleavedSamplerConfig(
                batch_size=4,
                drop_last=False,
                max_epochs=2,
            ),
        ),
        expected=[
            # main
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            # main
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            # configs[0]
            10,
            11,
            12,
            13,
            14,
        ],
    )


def test_sequential_nodroplast_enu1sequential():
    _run(
        sampler=InterleavedSampler(
            train_sampler=SequentialSampler(list(range(10))),
            callback_samplers=[
                SamplerIntervalConfig(
                    sampler=SequentialSampler(list(range(5))),
                    every_n_updates=1,
                    pipeline=None,
                ),
            ],
            config=InterleavedSamplerConfig(
                batch_size=4,
                drop_last=False,
                max_epochs=1,
            ),
        ),
        expected=[
            # main
            0,
            1,
            2,
            3,
            # configs[0]
            10,
            11,
            12,
            13,
            14,
            # main
            4,
            5,
            6,
            7,
            # configs[0]
            10,
            11,
            12,
            13,
            14,
            # main
            8,
            9,
            # configs[0] (last batch is counted as an update as drop_last=False)
            10,
            11,
            12,
            13,
            14,
        ],
    )


def test_sequential_droplast_enu1sequential():
    _run(
        sampler=InterleavedSampler(
            train_sampler=SequentialSampler(list(range(10))),
            callback_samplers=[
                SamplerIntervalConfig(
                    sampler=SequentialSampler(list(range(5))),
                    every_n_updates=1,
                    pipeline=None,
                ),
            ],
            config=InterleavedSamplerConfig(
                batch_size=4,
                drop_last=True,
                max_epochs=1,
            ),
        ),
        expected=[
            # main
            0,
            1,
            2,
            3,
            # configs[0]
            10,
            11,
            12,
            13,
            14,
            # main
            4,
            5,
            6,
            7,
            # configs[0]
            10,
            11,
            12,
            13,
            14,
        ],
    )


def test_sequential_droplast_enu1sequential_enu2sequential():
    _run(
        sampler=InterleavedSampler(
            train_sampler=SequentialSampler(list(range(10))),
            callback_samplers=[
                SamplerIntervalConfig(
                    sampler=SequentialSampler(list(range(5))),
                    every_n_updates=1,
                    pipeline=None,
                ),
                SamplerIntervalConfig(
                    sampler=SequentialSampler(list(range(2))),
                    every_n_updates=2,
                    pipeline=None,
                ),
            ],
            config=InterleavedSamplerConfig(
                batch_size=4,
                drop_last=True,
                max_epochs=1,
            ),
        ),
        expected=[
            # main
            0,
            1,
            2,
            3,
            # configs[0]
            10,
            11,
            12,
            13,
            14,
            # main
            4,
            5,
            6,
            7,
            # configs[0]
            10,
            11,
            12,
            13,
            14,
            # configs[1]
            15,
            16,
        ],
    )


def test_sequential_enu2sequential():
    _run(
        sampler=InterleavedSampler(
            train_sampler=SequentialSampler(list(range(10))),
            callback_samplers=[
                SamplerIntervalConfig(
                    sampler=SequentialSampler(list(range(5))),
                    every_n_updates=2,
                    pipeline=None,
                ),
            ],
            config=InterleavedSamplerConfig(
                batch_size=4,
                max_epochs=1,
                drop_last=False,
            ),
        ),
        expected=[
            # main
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            # configs[0]
            10,
            11,
            12,
            13,
            14,
            # main
            8,
            9,
        ],
    )


def test_sequential_nodroplast_ens8sequential():
    _run(
        sampler=InterleavedSampler(
            train_sampler=SequentialSampler(list(range(10))),
            callback_samplers=[
                SamplerIntervalConfig(
                    sampler=SequentialSampler(list(range(5))),
                    every_n_samples=8,
                    batch_size=4,
                    pipeline=None,
                ),
            ],
            config=InterleavedSamplerConfig(
                batch_size=4,
                drop_last=False,
                max_epochs=2,
            ),
        ),
        expected=[
            # main
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            # configs[0]
            10,
            11,
            12,
            13,
            14,
            # main
            8,
            9,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            # configs[0]
            10,
            11,
            12,
            13,
            14,
            # main
            8,
            9,
        ],
    )


def test_random_enu2sequential():
    _run(
        sampler=InterleavedSampler(
            train_sampler=RandomSampler(list(range(10)), generator=torch.Generator().manual_seed(0)),
            callback_samplers=[
                SamplerIntervalConfig(
                    sampler=SequentialSampler(list(range(5))),
                    every_n_updates=2,
                    pipeline=None,
                ),
            ],
            config=InterleavedSamplerConfig(
                batch_size=4,
                max_epochs=1,
                drop_last=False,
            ),
        ),
        expected=[
            # main
            4,
            1,
            7,
            5,
            3,
            9,
            0,
            8,
            # configs[0]
            10,
            11,
            12,
            13,
            14,
            # main
            6,
            2,
        ],
    )


def test_distsequential_enu1distsequential_rank0of2():
    _run(
        sampler=InterleavedSampler(
            train_sampler=DistributedSampler(ListDataset(list(range(10))), shuffle=False, num_replicas=2, rank=0),
            callback_samplers=[
                SamplerIntervalConfig(
                    sampler=DistributedSampler(ListDataset(list(range(5))), shuffle=False, num_replicas=2, rank=0),
                    every_n_updates=1,
                    pipeline=None,
                ),
            ],
            config=InterleavedSamplerConfig(
                batch_size=4,
                max_epochs=1,
                drop_last=False,
            ),
        ),
        expected=[
            # main
            0,
            2,
            4,
            6,
            # configs[0]
            10,
            12,
            14,
            # main
            8,
            # configs[0]
            10,
            12,
            14,
        ],
    )


def test_distsequential_enu1distsequential_rank1of2():
    _run(
        sampler=InterleavedSampler(
            train_sampler=DistributedSampler(ListDataset(list(range(10))), shuffle=False, num_replicas=2, rank=1),
            callback_samplers=[
                SamplerIntervalConfig(
                    sampler=DistributedSampler(ListDataset(list(range(5))), shuffle=False, num_replicas=2, rank=1),
                    every_n_updates=1,
                    pipeline=None,
                ),
            ],
            config=InterleavedSamplerConfig(
                batch_size=4,
                max_epochs=1,
                drop_last=False,
            ),
        ),
        expected=[
            # main
            1,
            3,
            5,
            7,
            # configs[0]
            11,
            13,
            10,
            # main
            9,
            # configs[0]
            11,
            13,
            10,
        ],
    )


def test_sequential_droplast_nofullepoch():
    _run(
        sampler=InterleavedSampler(
            train_sampler=SequentialSampler(list(range(100))),
            callback_samplers=[
                SamplerIntervalConfig(
                    sampler=SequentialSampler(list(range(5))),
                    every_n_samples=8,
                    batch_size=4,
                    pipeline=None,
                ),
                SamplerIntervalConfig(
                    sampler=SequentialSampler(list(range(5))),
                    every_n_epochs=1,
                    pipeline=None,
                ),
            ],
            config=InterleavedSamplerConfig(
                batch_size=4,
                drop_last=True,
                max_samples=16,
            ),
        ),
        expected=[
            # main
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            # config[0]
            100,
            101,
            102,
            103,
            104,
            # main
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            # config[0]
            100,
            101,
            102,
            103,
            104,
        ],
    )


def test_sequential_droplast_nointervalonsampleend():
    _run(
        sampler=InterleavedSampler(
            train_sampler=SequentialSampler(list(range(100))),
            callback_samplers=[
                SamplerIntervalConfig(
                    sampler=SequentialSampler(list(range(5))),
                    every_n_samples=6,
                    batch_size=4,
                    pipeline=None,
                ),
            ],
            config=InterleavedSamplerConfig(
                batch_size=4,
                drop_last=True,
                max_samples=16,
            ),
        ),
        expected=[
            # main
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            # config[0]
            100,
            101,
            102,
            103,
            104,
            # main
            8,
            9,
            10,
            11,
            # config[0]
            100,
            101,
            102,
            103,
            104,
            # main
            12,
            13,
            14,
            15,
        ],
    )
