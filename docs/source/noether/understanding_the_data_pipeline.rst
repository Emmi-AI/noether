Understanding the Data Pipeline
===============================

This document describes how data flows from files on disk to training batches, with a focus on CFD
aerodynamics datasets. It covers how point clouds are subsampled, what an epoch means when working with
large meshes, and how shuffling is organized.

For a hands-on walkthrough that shows how to configure and run an end-to-end training pipeline, see
the `tutorial README <https://github.com/Emmi-AI/noether/blob/main/tutorial/README.MD>`_.


From Disk to Batch
------------------

A single training step involves four stages:

1. **Dataset** -- loads individual tensors from ``.pt`` files on disk via ``getitem_*`` methods.
2. **Sample processors** -- transform each sample independently (subsample points, normalize, concatenate features).
3. **Collation** -- combines a list of processed samples into a batched dictionary of tensors.
4. **Batch processors** -- optional post-collation transforms on the full batch.

Stages 2--4 are orchestrated by :py:class:`~noether.data.pipeline.MultiStagePipeline`, which acts as
the ``collate_fn`` of the PyTorch ``DataLoader``.

.. code-block:: text

   Dataset.__getitem__(idx)
       |  torch.load() per property, @with_normalizers
       v
   Sample dict  {"surface_position": (N, 3), "volume_velocity": (M, 3), ...}
       |
       |  Sample processors (per sample, sequentially)
       v
   Processed sample dict  {"surface_position": (n, 3), ...}
       |
       |  Collation (list of dicts -> dict of tensors)
       v
   Batch dict  {"surface_position": (B, n, 3), ...}


The Decomposed ``getitem`` Pattern
-----------------------------------

Each dataset implements one ``getitem_<property>`` method per tensor it provides. The base
:py:class:`~noether.data.Dataset` discovers all such methods via introspection and calls them when
``__getitem__`` is invoked:

.. code-block:: python

   # simplified from base/dataset.py
   def __getitem__(self, idx):
       result = {"index": idx}
       for getitem_name in self.get_all_getitem_names():
           getitem_fn = getattr(self, getitem_name)
           result[getitem_name[len("getitem_"):]] = getitem_fn(idx)
       return result

By default, every ``__getitem__`` call invokes **all** ``getitem_*`` methods and therefore loads all
properties from disk. To avoid unnecessary I/O, we provide two ways to restrict which properties are
loaded:

- **Config-level**: set ``excluded_properties`` (or ``included_properties``) on the dataset config.
  This creates a :py:class:`~noether.data.base.wrappers.PropertySubsetWrapper` internally.
- **Code-level**: wrap the dataset explicitly with
  :py:class:`~noether.data.base.wrappers.PropertySubsetWrapper`.

For more details on implementing your own dataset, see :doc:`/guides/data/how_to_make_a_custom_dataset`.


Point Sampling
--------------

CFD meshes can contain tens of thousands to millions of points. To make training feasible, we
subsample points on-the-fly using
:py:class:`~noether.data.pipeline.sample_processors.PointSamplingSampleProcessor`:

.. code-block:: python

   PointSamplingSampleProcessor(
       items={"volume_position", "volume_velocity"},
       num_points=4096,
       seed=None,   # stochastic for training, fixed int for eval
   )

The processor generates a random permutation and takes the first ``num_points`` indices:

.. code-block:: python

   perm = torch.randperm(N, generator=generator)[: self.num_points]
   for item in self.items:
       output_sample[item] = output_sample[item][perm]

All items in the ``items`` set share the same permutation, so position-to-field correspondence is
preserved (``position[i]`` always matches ``velocity[i]``).

When ``seed`` is ``None``, a fresh random permutation is drawn every call. When ``seed`` is an
integer, the processor uses ``sample_index + seed`` to produce a deterministic permutation per
sample, ensuring reproducible evaluation.


Two Levels of Subsampling
~~~~~~~~~~~~~~~~~~~~~~~~~

Some datasets apply subsampling at two stages:

1. **Preprocessing time** (offline, one-off): for datasets with very large raw meshes (e.g. AhmedML
   and DrivAerML with millions of cells), the preprocessing script reduces the stored point count by
   a ``subsample_factor`` (default 10). This uses a deterministic permutation seeded by the
   simulation run ID.

2. **Training time** (online, every access): ``PointSamplingSampleProcessor`` further subsamples the
   stored points to the target count (e.g. 4096).

For datasets with smaller meshes (e.g. ShapeNet Car, where the volume has ~28k points), only
training-time subsampling is used.

For details on writing custom sample processors, see
:doc:`/guides/data/how_to_write_a_sample_processor`.


Epochs and Subsampling
----------------------

An epoch is one complete pass over the **samples** in the training dataset -- not over
individual points within those samples.

.. code-block:: python

   # InterleavedSampler.__init__
   if drop_last:
       samples_per_epoch = len(main_sampler) // batch_size * batch_size
   else:
       samples_per_epoch = len(main_sampler)

With ``drop_last=True`` (the default), the last incomplete batch is discarded, so
``samples_per_epoch`` is the largest multiple of ``batch_size`` that fits.

Point-level coverage
~~~~~~~~~~~~~~~~~~~~

Because ``PointSamplingSampleProcessor`` uses ``seed=None`` during training, each sample receives a
different random subset of points every epoch. This has two consequences:

- **Implicit data augmentation**: the model never sees exactly the same point set twice for a given
  sample. Across epochs, it learns from many overlapping views.

- **No systematic coverage**: there is no mechanism that tracks which points have been visited. For a
  sample with 28,504 volume points sampled at 4,096 per epoch, each point has a ~14.4% chance of
  being selected per epoch (4096/28504). After *E* epochs the expected coverage is
  ``1 - (1 - 4096/28504)^E``, which gives ~87% after 13 epochs and ~95% after about 19 epochs.

In other words, an epoch guarantees that every **sample** is visited once, but says nothing about
which **points** within each sample are selected.

RepeatWrapper for robust evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since point subsampling is stochastic, a single evaluation pass may not give a stable estimate of
model performance. :py:class:`~noether.data.base.wrappers.RepeatWrapper` addresses this by tiling
the dataset indices *N* times:

.. code-block:: python

   RepeatWrapper(config=RepeatWrapperConfig(repetitions=10), dataset=test_dataset)
   # 100 test samples -> 1000 (each sample visited 10 times)

When the pipeline's ``seed`` is ``None`` (the default), each sample is evaluated with 10 different
random point subsets, giving a more robust average. This is typically run less frequently (e.g., every
500 epochs) to amortize the cost.


Shuffling
---------

Data ordering is controlled at two independent levels.

Dataset-level shuffle
~~~~~~~~~~~~~~~~~~~~~

:py:class:`~noether.data.base.wrappers.ShuffleWrapper` applies a one-time, fixed permutation of
sample indices at dataset construction time. This does **not** re-shuffle between epochs. Its typical
use is to randomize order before taking a deterministic subset (e.g., shuffle, then slice the first
80% for training).

Sampler-level shuffle
~~~~~~~~~~~~~~~~~~~~~

The ``DataContainer`` creates a ``RandomSampler`` (single-GPU) or ``DistributedSampler``
(multi-GPU) as the main sampler. These produce a fresh random ordering of indices every epoch.

For ``DistributedSampler``, the :py:class:`~noether.data.samplers.InterleavedSampler` calls
``set_epoch(epoch)`` at the start of each epoch to re-seed the shuffle. For ``RandomSampler``,
calling ``iter()`` produces a new permutation.


Interleaved Sampling
--------------------

A typical PyTorch training loop uses separate ``DataLoader`` instances for training and evaluation.
The Noether Framework takes a different approach: it uses a **single DataLoader** that serves both
training and evaluation batches. We call this *interleaved sampling*.

How it works
~~~~~~~~~~~~

The :py:class:`~noether.data.samplers.InterleavedSampler` concatenates all datasets (train, test,
test_repeat, etc.) into one ``ConcatDataset``. It then produces a stream of index tuples from a
single iterator:

1. Most of the time, it yields indices from the main (training) sampler (e.g. ``RandomSampler``).
2. When a configured interval is reached (e.g. end of an epoch), it **pauses** the training sampler,
   yields **all** indices from a callback sampler (e.g. ``SequentialSampler`` over the test set),
   and then **resumes** training where it left off.

Because all datasets live in one ``ConcatDataset``, callback indices are offset so they map to the
correct sub-dataset. For example, if the training set has 789 samples, test index 0 becomes global
index 789. An ``_InterleavedCollator`` inspects which sub-dataset each batch comes from and applies
the corresponding pipeline (collation function).

.. code-block:: python

   # The training loop just iterates -- the sampler handles the rest:
   for batch in interleaved_loader:
       # Most batches come from the training set.
       # Periodically, evaluation batches are interleaved in.
       process(batch)

Configuring callbacks
~~~~~~~~~~~~~~~~~~~~~

Each callback sampler is wrapped in a
:py:class:`~noether.data.samplers.SamplerIntervalConfig` that specifies when it should run:

.. code-block:: python

   SamplerIntervalConfig(
       sampler=SequentialSampler(test_dataset),
       pipeline=eval_pipeline,
       every_n_epochs=1,
   )

Intervals can be epoch-based, update-based, or sample-based (mutually exclusive per callback).
Each callback can use its own ``pipeline`` and ``batch_size``.

Training stops when any configured limit is reached first: ``max_epochs``, ``max_updates``, or
``max_samples``. Resume is supported via ``start_epoch``, ``start_update``, or ``start_sample``,
including mid-epoch resume.


Loading and Performance
-----------------------

Every ``__getitem__`` call reads the full tensor from disk via ``torch.load``. There is currently no
caching, memory-mapping, or partial-read support. For a ShapeNet Car volume property, this means
loading ~28k points to ultimately keep 4,096 after point sampling.

The :py:class:`~noether.data.pipeline.MultiStagePipeline` deep-copies every sample before applying
processors, so the original dataset tensors are never mutated.

Several mechanisms help keep loading efficient in practice:

- **Property filtering** via ``excluded_properties`` prevents loading tensors that are not needed for
  the current training run.
- **DataLoader workers**: with ``num_workers > 0``, loading runs in parallel subprocesses overlapped
  with GPU computation. The default is ``(#CPUs / #GPUs) - 1`` workers.
- **Preprocessing-time subsampling**: for CAEML datasets, the stored files are already 10x smaller
  than the raw mesh thanks to the offline ``subsample_factor``.
- **File sizes**: for ShapeNet Car, individual ``.pt`` files are under 1 MB each, so per-read cost is
  modest.

The design prioritizes simplicity and flexibility -- any sample processor can operate on the full
point cloud -- over minimal I/O.


Dataset Statistics
------------------

Normalizing inputs and targets requires dataset-wide statistics (mean, standard deviation, min, max).
We compute these in a single pass using Welford's online algorithm via
:py:class:`~noether.data.stats.RunningMoments`, which operates in float64 for numerical stability
and avoids loading the entire dataset into memory at once.

The ``noether-dataset-stats`` CLI automates this:

.. code-block:: bash

   noether-dataset-stats \
       --dataset_kind=noether.data.datasets.cfd.shapenet_car.ShapeNetCarDataset \
       --split=train \
       --root=/path/to/data \
       --exclude_attributes=surface_friction,volume_pressure,volume_vorticity

The tool sets ``compute_statistics=True`` on the dataset, which disables ``@with_normalizers`` so
that raw (unnormalized) values are processed. The output statistics are then used to populate
normalizer configs for training.
