How to Implement a Custom Sample Processor 
========================================

Inside the Multistage Pipeline, data samples can be processed using custom Sample Processors.
A Sample Processor is a callable class that takes as input a single data sample (a dictionary) and returns a processed data sample (also a dictionary).
To create your own Sample Processor, you need to extend the base ``SampleProcessor`` class and implement the ``__call__`` method.
Sample processors do not receive a configuration object, but can accept arbitrary keyword arguments in their constructor.

.. code-block:: python

   from noether.data.pipeline.sample_processor import SampleProcessor
   
   class CustomSampleProcessor(SampleProcessor):
      """Utility processor that simply duplicates the dictionary keys in a batch."""

      def __init__(self, **kwargs) -> None:
         """
         Args:
            Sample processor don't get a config object as input, but can accept arbitrary keyword arguments.
         """

       

      def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
         """
         Args:
             input_sample: Input sample dictionary.
         Returns:
               Processed sample dictionary.
         """
         
         # do any form of processing here
      
         return output_sample
