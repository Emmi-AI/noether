Getting Started: Install & Verify
=================================

To use the **Noether Framework** you have to clone the repository from GitHub or install it as a Python package.
This documentation will guide you through this process.

We assume basic knowledge of Git, Python, as well as Unix systems.

Installing from the repo
------------------------

To get started, clone the repo to your machine:

.. code-block:: bash

   git clone git@github.com:Emmi-AI/noether.git

Navigate to your fresh repo, here we will assume the working path as `/home/user/repos/noether`, and use `uv` to set up
the project.

Create a virtual environment with relevant packages:

.. code-block:: bash

    uv venv                    # create a default virtual environment under the default .venv folder
    source .venv/bin/activate  # activate your newly created environment in your terminal
    uv sync                    # install project dependencies

.. note::

    If you installing Noether for the first time, the initial build time can take a few minutes. In the follow-up
    setups ``uv`` will cache packages and installation will be much faster.

Verifying the Installation
--------------------------

See if the installation was successful by running the ``noether-train`` :

.. code-block:: bash

   python -c "uv run noether-train --help"

You should see the help message printed in your terminal.