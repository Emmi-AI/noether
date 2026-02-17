How-to Guides
=============

Practical guides for setting up your environment, managing data, and customizing training and inference.

Setup & Environment
-------------------

* :doc:`linux_cuda_setup`: Step-by-step instructions for configuring Linux and CUDA.
* :doc:`working_with_cli`: How to use Noether's command-line interface tools.
* :doc:`training/experiment_tracking`: Integrating experiment tracking tools like Weights & Biases or trackio.

Data Management
---------------

* :doc:`data/how_to_use_private_data_source`: Configuring private data storage like AWS S3.
* :doc:`data/how_to_make_a_custom_dataset`: A guide to implementing your own dataset classes.
* :doc:`data/how_to_write_a_multistage_pipeline`: Designing complex, multi-step data pipelines.
* :doc:`data/how_to_write_a_sample_processor`: Customizing how individual data samples are processed.

Customizing Training
--------------------

* :doc:`training/implement_a_custom_model`: Defining new model architectures and layers.
* :doc:`training/implement_a_custom_trainer`: Tailoring the training loop for specific research or production needs.
* :doc:`training/use_callbacks`: Extending the framework with custom training callbacks.

Inference & Evaluation
----------------------

* :doc:`inference/how_to_run_evaluation_on_trained_models`: Running pre-trained models on evaluation sets.

.. toctree::
   :maxdepth: 1
   :hidden:

   linux_cuda_setup
   working_with_cli

   data/how_to_use_private_data_source
   data/how_to_make_a_custom_dataset
   data/how_to_write_a_multistage_pipeline
   data/how_to_write_a_sample_processors
   
   training/implement_a_custom_model
   training/implement_a_custom_trainer
   training/use_callbacks
   training/experiment_tracking

   inference/how_to_run_evaluation_on_trained_models
