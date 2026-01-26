How to Implement a Custom Model
===============================

You have to extend the :py:class:`noether.core.models.Model` class and implement the ``__init__`` and :py:meth:`noether.core.models.Model.forward` methods
Additionally, you will need to create a corresponding model configuration class that extends :py:class:`noether.core.schemas.models.ModelBaseConfig` to define the model-specific parameters.


.. code-block:: python

    from noether.core.models import Model
    from noether.core.schemas.models import ModelBaseConfig

    class CustomModelConfig(ModelBaseConfig):
        input_dim: int
        hidden_dim: int
        output_dim: int

    class CustomModel(Model):
        def __init__(self, model_config: CustomModelConfig, **kwargs):
            # the model config needs to be passed to the parent Model class 
            super().__init__(model_config=model_config, **kwargs)
        
            self.config = model_config
            
            # Define your model layers here
            self.encoder = torch.nn.Linear(model_config.input_dim, model_config.hidden_dim)
            self.decoder = torch.nn.Linear(model_config.hidden_dim, model_config.output_dim)
        
        def forward(self, input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
            """
            Forward pass of the model.
            
            Args:
                input_tensor: torch tensor with data 
            
            Returns:
                Dictionary containing model outputs.
            """
            # Example: extract inputs from batch
            x = input_tensor
            
            # Forward pass
            hidden = self.encoder(x)
            output = self.decoder(hidden)
            
            return {'output':output}

An example configuration for this custom model in YAML format would look like this:

.. code-block:: yaml

    kind: path.to.CustomModel
    name: custom_model
    input_dim: 3
    hidden_dim: 128
    output_dim: 1
    optimizer_config: ${optimizer}  # Reference to optimizer defined elsewhere
    forward_properties:
      - input_tensor
