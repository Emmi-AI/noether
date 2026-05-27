Noether Recipe Zoo
==================

All recipe source code lives under the `recipes/ <https://github.com/Emmi-AI/noether/tree/main/recipes>`_ directory.

.. grid:: 1
   :gutter: 3
   :class-container: sd-mb-3

   .. grid-item-card:: :ph:`car-profile` AB-UPT Showcase on DrivAerML
      :img-top: /_static/showcase/inference_scaling.png
      :class-card: sd-shadow-lg sd-rounded-3
      :class-title: sd-fs-4 sd-fw-bold sd-text-primary
      :class-body: sd-p-4

      :bdg-success:`Featured`

      AB-UPT on the DrivAerML automotive aerodynamics benchmark with a preset-based CLI for training,
      evaluation, and VTK export.

      - :doc:`Documentation </tutorials/walkthrough/showcase>`
      - `Source code <https://github.com/Emmi-AI/noether/tree/main/recipes/aero_cfd/showcase>`_

.. grid:: 1 2 2 2
   :class-container: sd-equal-height

   .. grid-item-card:: :ph:`wind` External Aerodynamics
      :img-top: /_static/showcase/750k_points_inference_drivaerml_sample_0000.png
      :class-card: sd-h-100 sd-shadow-sm sd-rounded-3
      :class-title: sd-fs-5 sd-fw-bold sd-text-primary
      :class-body: sd-p-4
      :class-img-top: recipe-card-thumb

      Multi-dataset aero CFD (ShapeNet-Car, AhmedML, DrivAerML, DrivAerNet++, Emmi-Wing) across AB-UPT, UPT,
      Transformer, and Transolver architectures.

      - :doc:`Full walkthrough </tutorials/walkthrough/index>`
      - :doc:`Python scripts reference <aero_cfd_python>`
      - `Source code <https://github.com/Emmi-AI/noether/tree/main/recipes/aero_cfd>`_

   .. grid-item-card:: :ph:`thermometer-hot` Fluid Heat Transfer
      :img-top: /_static/heat_transfer/temperature_prediction.png
      :class-card: sd-h-100 sd-shadow-sm sd-rounded-3
      :class-title: sd-fs-5 sd-fw-bold sd-text-primary
      :class-body: sd-p-4
      :class-img-top: recipe-card-thumb

      Neural surrogates for heat transfer on the SIMSHIFT Heatsink benchmark -- predicts 3D velocity,
      temperature, and pressure fields from heatsink geometry.

      - :doc:`Documentation <heat_transfer>`
      - `Source code <https://github.com/Emmi-AI/noether/tree/main/recipes/heat_transfer>`_

   .. grid-item-card:: :ph:`chart-line` Uncertainty Quantification
      :img-top: /_static/uncertainty_quantification/friction_uq.png
      :class-card: sd-h-100 sd-shadow-sm sd-rounded-3
      :class-title: sd-fs-5 sd-fw-bold sd-text-primary
      :class-body: sd-p-4
      :class-img-top: recipe-card-thumb

      Aleatoric uncertainty estimates for AB-UPT on DrivAerML. 

      - :doc:`Documentation <uncertainty_quantification>`
      - `Source code <https://github.com/Emmi-AI/noether/tree/main/recipes/uncertainty_quantification>`_

.. toctree::
   :hidden:

   aero_cfd_python
   heat_transfer
   uncertainty_quantification
