How to Implement a Custom Multistage Pipeline
=============================================

Within the `Noether Framework`, data samples loaded by the `Dataset` are processed through a Multistage Pipeline that consists of three stages: sample preprocessors, collators, and batch postprocessors.
To create your own Multistage Pipeline, you need to extend the base :py:class:`~noether.data.pipeline.MultiStagePipeline` class and provide your custom processors and collators in the constructor.

You have to provide three lists to the `__init__` of the partent class: one for sample preprocessors, one for collators, and one for batch postprocessors.
In the preprocessors list, instances of :py:class:`~noether.data.pipeline.sample_processor.SampleProcessor` classes are included that process individual data samples.
In the collators list, instances of :py:class:`~noether.data.pipeline.collator.Collator` classes are included that collate the list of samples into batches.
In the postprocessors list, :py:class:`~noether.data.pipeline.batch_processor.BatchProcessor` classes are included that process the entire batch after collation.
Those lists can also be empty if you do not wish to include any processors or collators in that stage.

Below we provide a minimal (dummy code) example of how to create a custom Multistage Pipeline by extending the base :py:class:`~noether.data.pipeline.MultiStagePipeline` class.
The sample processors we implemented do not take a config object as input, but can accept arbitrary keyword arguments in their constructor.

.. code-block:: python

   from noether.data.pipeline import MultiStagePipeline
   from own_project.data.pipeline.sample_processors import SampleProcessorA, SampleProcessorB, SampleProcessorC
   from own_project.data.pipeline.batch_processors import BatchProcessor
   from noether.data.pipeline.collators import  DefaultCollator 
   
   class CustomMultiStagePipeline(MultiStagePipeline):
      def __init__(self, pipeline_config: CustomMultiStagePipelineConfig, **kwargs) -> None:

         super().__init__(
            preprocessors=[
               SampleProcessorA(**pipeline_config.sample_processor_a_config), 
               SampleProcessorB(**pipeline_config.sample_processor_b_config), 
               SampleProcessorC(**pipeline_config.sample_processor_c_config)
            ],
            collators=[DefaultCollator(**pipeline_config.default_collator_config), CustomCollator(**pipeline_config.custom_collator_config)],
            postprocessors=[BatchProcessor(**pipeline_config.batch_processor_config)],
            **kwargs,
         )

