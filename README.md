<div align="center">

![noether-framework-logo](./docs/source/_static/noether-framework-logo.svg)
# Noether Framework

[![Docs - noether-docs.emmi.ai](https://img.shields.io/static/v1?label=Docs&message=noether-docs.emmi.ai&color=2ea44f&logo=gitbook)](https://noether-docs.emmi.ai)
[![License: ENLP](https://img.shields.io/badge/License-ENLP-orange.svg)](./LICENSE.txt)
[![Static Badge](https://img.shields.io/badge/Walkthrough-Tutorial-DD537C)](./tutorial/README.MD)

[![Tests](https://github.com/Emmi-AI/noether/actions/workflows/run-tests.yml/badge.svg)](https://github.com/Emmi-AI/noether/actions/workflows/run-tests.yml)

</div>

<div>
<strong>Noether</strong> is Emmi AI’s <strong>open software framework for Engineering AI</strong>. Built on 
<strong>transformer building blocks</strong>, it delivers the full engineering stack, allowing teams to build, 
train, and operate industrial simulation models across engineering verticals, eliminating the need for component 
re-engineering or an in-house deep learning team.
</div>

## Key Features

- **Modular Transformer Architecture:** Built on building blocks optimized for physical systems.
- **Hardware Agnostic:** Seamless execution across CPU, MPS (Apple Silicon), and NVIDIA GPUs.
- **Industrial Grade:** Designed for high-fidelity industrial simulations and engineering verticals.
- **Ready for Scale:** Built-in support for Multi-GPU and SLURM cluster environments.

---

# Table of contents

- [Installation](#installation)
  - [Pre-requisites](#pre-requisites)
  - [Working with the source code](#working-with-the-source-code)
    - [How to clean up and do a fresh installation](#how-to-clean-up-and-do-a-fresh-installation)
  - [Working with pre-built packages](#working-with-pre-built-packages)
- [Quickstart](#quickstart)
- [Performance Benchmarks](#performance-benchmarks)
- [Contributing](#contributing)
  - [Guidelines](#guidelines)
  - [Third-party contributors](#third-party-contributors)
  - [Configuring IDEs](#configuring-ides)
- [Supported systems](#supported-systems)
- [Licensing](#licensing)
- [Endorsements](#endorsed-by)
- [Citing](#citing)

---
# Installation

It is possible to use the framework either from source or from the pre-built packages.

## Pre-requisites

- install [uv](https://docs.astral.sh/uv/getting-started/installation/) as the package manager on your system
- clone the repo into your desired folder: `git clone https://github.com/Emmi-AI/noether.git`
- follow the next steps 🚀

## Working with pre-built packages

Installable package is available via `pip` and can be installed as:

```bash
pip install emmiai-noether
```

we recommend using `uv` as your package manager for better dependencies resolution, you can use it like this:

```bash
uv add emmiai-noether
```
or
```bash
uv pip install emmiai-noether
```

## Working with the source code

If you prefer to work with the source code directly without installing a prebuilt package.

> [!IMPORTANT]
> If you are running on NVIDIA GPUs or need custom CUDA paths, you must configure your environment variables first. 
> Please follow our [Advanced Linux Setup Guide](https://noether-docs.emmi.ai/guides/linux_cuda_setup.html) before 
> running the command below.

Create a fresh virtual environment and synchronize the core dependencies:

```console
uv venv && source .venv/bin/activate
uv sync
```

**Note:** Initial installation may take several minutes as third-party dependencies are compiled. Duration depends on 
your hardware and network speed.

Validate your installation by simply running the tests (if something fails with module import errors it means that the 
installation was incomplete):
```console
pytest -q tests/
```
if the tests are passed (warnings are okay to be logged) then you're all set and ready to go!

### How to clean up and do a fresh installation

You might be in a situation when your venv won't be configured as intended anymore, to fix this:

- Deactivate existing environment in your terminal by running: `deactivate`
- Remove existing `.venv` (optionally add `uv.lock`): `rm -rf .venv uv.lock`
- [Optional] Clean uv cache: `uv cache clean`
- Create a new venv and activate it: `uv venv && source .venv/bin/activate`
- [Optional] If deleted, generate a new `uv.lock` file: `uv lock`
- [Optional] If contributor: `pre-commit install`

---
# Quickstart

You can run a training job immediately using the [tutorial](./tutorial/README.MD) configuration. For local development 
(Mac/CPU), use:

```console
uv run noether-train --hp tutorial/configs/train_shapenet.yaml \
    +experiment/shapenet=upt \
    dataset_root=./data \
    +accelerator=mps \
```

Learn more about different hardware support [here](https://noether-docs.emmi.ai/guides/hardware_setup.html).

---
# Performance Benchmarks

The following benchmarks demonstrate **Noether**'s acceleration compatibility over different hardware using 
the `ShapeNet-Car` dataset and the `AB-UPT` model. 

> [!NOTE]
> All benchmarks were conducted using **FP32 precision** to establish a baseline for raw computational performance.

| **Hardware**                 | **Config** | **Precision** | **Time** | **Speedup** |
|:-----------------------------|:-----------|:--------------|:---------|:------------|
| **MacBook Pro M3 Max**       | 1x MPS     | FP32          | 135m     | 1.0x        |
| **RTX Pro 4500 (Blackwell)** | 1x GPU     | FP32          | 26m      | 5.2x        |
| **RTX Pro 4500 (Blackwell)** | 2x GPU     | FP32          | 8m       | 16.8x       |
| **NVIDIA H100**              | 1x GPU     | FP32          | 5.7m     | 23.6x       |

---
# Contributing

## Guidelines

We follow these standards:

- Use typed coding in Python.
- Write documentation to new features and modules:
  - In case of larger modules make sure to update the documentation that is not autogenerated under the `docs/`.
  - For smaller features writing a clear API documentation is enough and required.
- Before committing your changes:
  - Run tests via `pytest -q tests/`.
  - Ensure that [pre-commit hooks](#pre-commit-hooks) are not disabled and are runnable at every commit.
    We are using `ruff` as a linter and formatter as well as `mypy` for type checking.
    Their configuration is defined in the project's root [pyproject.toml](pyproject.toml).
- Creating pull requests (PRs) is a mandatory step for any incoming changes that will end up on the `main` branch.
  - For a PR to be merged at least one core maintainer must give their approval.
  - All test must be green

## Pre-commit Hooks

To install pre-commit execute:
```console
pre-commit install
```
To run the pre-commit configuration on all files, you can use:
```console
pre-commit run --all-files
```
To run the pre-commit configuration on specific files use:
```console
pre-commit run --files /your/file/path1.py /your/file/path2.py
```

## Third-party contributors

In case of bugs use a corresponding template to create an issue.

In case of feature requests you can submit a PR with clear description of the proposed feature. In that case it must 
follow the [guidelines](#guidelines), or file a feature request as an issue. In that case, we will consider adding it to our 
backlog.

## Configuring IDEs

### Pycharm

- Mark `src/` directory as `Sources Root` (right mouse button click on the folder -> `Mark Directory as`)
- Settings -> Editor -> Code Style -> Python -> Tabs and Indents -> change `Continuation indent` from 8 to 4.
- Settings -> Editor -> Code Style -> Python -> Spaces -> Around Operators -> `Power operator (**)`

---
# Working with GitHub

With available GitHub Actions we automate several workflows relevant to our development ranging from buildings the docs
to building our modules as wheel files.

To test the desired workflow locally it is recommended to use [act](https://github.com/nektos/act).

> [!NOTE]
> Make sure to install Docker Desktop as requested by the official documentation.

Install it on a Mac with: `brew install act`

For example, to check the package release pipeline:
```console
act workflow_dispatch --input version_type=patch -W .github/workflows/release.yml
```
or to see if tests are runnable:
```console
act pull_request -W .github/workflows/run-tests.yml
```

---
# Supported systems
Worth noting that we work with macOS and Linux environments thus in case of any issues on Windows, at this time, you 
have to find workarounds yourself.

---
# Licensing

> [!NOTE]
> TL;DR: Research & development ✅| Production deployment ❌ (without commercial license)

The Noether Framework is licensed under a Non-Production License (based on Mistral AI's MNPL). This means you're free 
to use, modify, and research with the framework, but commercial/production use requires a separate commercial license 
from Emmi AI.

We're committed to open AI innovation while sustainably growing our business. For commercial licensing, contact 
us at partner@emmi.ai .

Read the full license [here](./LICENSE.txt).

---
# Endorsed by research groups from

<table>
  <tr>
    <td align="center">
      <img src="docs/source/_static/logos_cards/jku.png" height="50" alt="JKU Linz">
    </td>
    <td align="center">
      <img src="docs/source/_static/logos_cards/eth.png" height="50" alt="ETH Zurich">
    </td>
    <td align="center">
      <img src="docs/source/_static/logos_cards/upenn.png" height="50" alt="UPenn">
    </td>
    <td align="center">
      <img src="docs/source/_static/logos_cards/uw.png" height="50" alt="University of Washington">
    </td>
    <td align="center">
      <img src="docs/source/_static/logos_cards/tum.png" height="50" alt="TUM Munich">
    </td>
    <td align="center">
      <img src="docs/source/_static/logos_cards/sorbonne.png" height="50" alt="Sorbonne University">
    </td>
  </tr>
</table>

---
# Citing

If you use **Noether** in your research or industrial applications, please cite this repository. 
A formal BibTeX entry for our forthcoming ArXiv publication will be provided here shortly.

```bibtex
@misc{noether2026,
  author = { Bleeker, Maurits AND Hennerbichler, Markus AND Kuksa, Pavel },
  title = {Noether: A PyTorch-based Framework for Engineering AI},
  year = {2026},
  publisher = {GitHub},
  note = {Equal contribution},
  url = {https://github.com/Emmi-AI/noether}
}
```
