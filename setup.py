from setuptools import find_packages, setup

setup(
    name='scaleup_optimizer_libs',
    packages=find_packages(),
    version='0.1.0',
    description='This library is use to optimize hyperparameter of machine learning with scale up algorithm',
    author='Ly Sreypov',
    install_requires=['numpy>=1.21.0', 'scipy>=1.10.0', 'scikit-optimize>=0.8.1']
)