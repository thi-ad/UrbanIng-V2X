# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


from os.path import dirname, realpath
from setuptools import setup, find_packages, Distribution
from opencood.version import __version__


def _read_requirements_file():
    """Return the elements in requirements.txt."""
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name='UrbanIng-V2X',
    version=__version__,
    packages=find_packages(),
    url='https://github.com/thi-ad/UrbanIng-V2X.git',
    license='MIT',
    author='Dominik Rößle',
    author_email='dominik.roessle@thi.de',
    description='UrbanIng-V2X integration for OpenCOOD. !The authors of the original implementation are Runsheng Xu and Hao Xiang!',
    long_description=open("README.md").read(),
    install_requires=_read_requirements_file(),
)
