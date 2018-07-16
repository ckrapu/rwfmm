# RWFMM
This package implements several types of functional mixed model in PyMC3 and
provides high-level functions to fit them to data using Markov chain Monte Carlo
or variational inference. The intended use cases consist of regression continuous
outputs on longitudinal measurements of scalar and 1D functional data. We
currently include two different ways to specify the functional coefficients:
a random walk across the function domain and a B-spline basis for the coefficient.

## Getting Started
This software is still in its preliminary stages. In order to install it
locally on a Linux machine, run `pip install git+git://github.com/ckrapu/rwfmm.git`.

### Prerequisites

The prerequisites are PyMC3 and Matplotlib. Numpy and Theano are also used, but
should already be installed with PyMC3.


## Authors

* **Chris Krapu** **Drew Day**


## License

This project is licensed under the MIT License.
