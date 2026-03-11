How to launch a SLURM job from the command line
===============================================

Via the ``noether-train-submit-job`` command, you can launch a training job directly from the command line. 
You can specify the path to your config file, override any config options.
The script will first validate the config, then create a SLURM job command and submit it to the cluster.

For example: 

.. code-block:: bash

   noether-train-submit-job /path/to/noether/tutorial/configs/train_shapenet.yaml  +experiment/shapenet=transformer +seed=1 tracker=disabled dataset_root=/path/to/datasets/shapenet_car/

The SLURM variables used for the job (e.g., number of GPUs, number of CPUs, etc.) have to be defined in the config schema under the ``slurm`` key. 

An example of the SLURM config in YAML looks like this:

.. code-block:: yaml

   slurm:
      nodes: 1
      cpus_per_task: 28
      partition: compute
      gpus_per_node: 1
      ntasks_per_node: 1
      mem: 64GB
      output: /home/%u/logs/shapenet_car/%x_%j.out
      nice: 0
      job_name: shapenet_experiment
      chdir: # optional, if not set, the job will be launched from the current working directory
      env_path: # path to a environment file to source before launching the job, e.g., /home/user/.bashrc could be absolute or relative the to chdir path

See the ShapeNet-Car config in the tutorial for a full example to setup the SLURM config. 