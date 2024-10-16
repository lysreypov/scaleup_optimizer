# scaleup_optimizer_libs

`scaleup_optimizer_libs` is a Python library for hyperparameter optimization using scalable algorithms. It integrates Bayesian optimization techniques to handle both small-scale and large-scale machine learning models.

## Description

This library is designed to help you efficiently optimize hyperparameters of machine learning models. It provides tools for working with both small-scale (e.g., experimental) and large-scale (e.g., production) systems, leveraging Bayesian optimization and Gaussian Process models to improve the performance of models at different stages of development.

Key features:
- Bayesian optimization with Expected Improvement (EI), Probability of Improvement (PI), and Upper Confidence Bound (UCB) acquisition functions.
- Support for different surrogate models like small-scale and large-scale Gaussian Processes.
- Customizable search space for hyperparameter optimization.
- Scalability to handle both experimental and production systems.

## Installation

### scaleup_optimizer_libs requires

- `numpy>=1.21.0`
- `scipy>=1.10.0`
- `scikit-optimize>=0.8.1`

### Install via pip

You can install the library from PyPI or your local environment:

#### From PyPI
```bash
pip install scaleup_optimizer_libs
