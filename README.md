# hyperion
Light-weight photon propagation implemented in jax.

## Quickstart
`examples/photon_propagation.ipynb shows an example of how to propagate photons.


## Getting Started

This section tells you how to set up the package and be able to run with it.

### Prerequisites

This package is built and updated using [Poetry](https://python-poetry.org/). 
Please install it and make yourself familiar if you never heard of it.

This package has been optimised for python 3.8, but 3.10 should work as well.

### Installation

To install the virtual environment of the package call the following console command

```console
foo@bar:hyperion/$ poetry install
```

### Using Jax and Torch

As we do not know whether you have a gpu or which cuda/python version you have, you
should install jax and torch manually in the environment. To do that enter the 
environment

```console
foo@bar:hyperion/$ poetry shell
```

And then install jax and torch as given on the homepages:

* [JAX](https://jax.readthedocs.io/en/latest/installation.html)
* [PyTorch](https://pytorch.org/)