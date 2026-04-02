Noether Development CLI
========================

The Noether Development CLI is a command-line interface design to develop the individual components of the Noether Framework in isolation. 
However, although modules can be developed in isolation, some modules still depend on each other in the larger framework.

Here is how the dependency graph of the main Noether modules looks like:

- 1. **Dataset**: The dataset is resposible from loading the data per sample from disk.
    - 2. **Pipeline**: The pipeline (i.e., collator) collates the data loaded by the dataset into a batch and applies additional transformations to the data (if applicable). The pipeline depends on the dataset and cannot be configured without one.
        - 3. **Model**: 
            - 4.1 **Trainer**: 
            - 4.2 **Callbacks** (For now only PeriodicDataIteratorCallbacks are supported)

  
.. code-block:: yaml 

    config_schema_kind: development.development_schema.DevelopmentSchema
    datasets:
    development_dataset:
        kind: development.datasets.DevelopmentDataset
        x_dim: 3
        y_dim: 2
        z_dim: 1                                                                                    
        sample_size: 10
        num_samples: 1000
        split: train  


.. code-block:: yaml 
    
    development_datset:
        ...
        dataset_normalizers:
            x:
                - kind: noether.data.preprocessors.normalizers.MeanStdNormalization
                  mean: [0, 0, 0]
                  std: [1, 1, 1]
            y:
                - kind: noether.data.preprocessors.normalizers.MeanStdNormalization
                  mean: [0, 0]
                  std: [1, 1]
            z:  
                - kind: noether.data.preprocessors.normalizers.MeanStdNormalization
                  mean: [0]
                  std: [1]

.. code-block:: yaml 

    development_datset:
        ...
        pipeline:
            kind: development.pipeline.DevelopmentPipeline

Pipeline needs to be present. 
.. code-block:: yaml

    ... 
    model: # can be commented out to test behavior when no model is defined, skip model instantiation and forward pass
        kind: development.model.DevelopmentModel
        input_dim: 5
        output_dim: ${datasets.development_dataset.y_dim}
        forward_properties: ["x_z"]