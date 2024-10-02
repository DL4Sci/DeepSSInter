#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='DeepSSInter',
    version='1.0.0',
    description='A deep learning model for predicting inter-protein contacts using structure-aware single-sequence protein language models.',
    author='anonymity',
    license='GNU Public License, Version 3.0',
    url='https://github.com/BioinfoMachineLearning/DeepInteract',
    install_requires=[
        'setuptools==57.4.0',
        'dill==0.3.4',
        'torchmetrics==0.5.1',
        'pytorch-lightning==1.4.8',
        'fair-esm==2.0.0',
        'tensorboard==2.14.0',
        'tensorboard-data-server==0.7.2',
        'wandb==0.12.2',
        'protobuf==3.20',
    ],
    packages=find_packages(),
)
