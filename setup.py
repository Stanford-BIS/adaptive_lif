#!/usr/bin/env python
import imp
import os

try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import find_packages, setup

root = os.path.dirname(os.path.realpath(__file__))
description = ("ALIF neurons for Nengo whose decoders are optimized for the" 
              " steady state response")
with open(os.path.join(root, 'README.md')) as readme:
    long_description = readme.read()

setup(
    name="nengo_alif_steady_state",
    version=1.0,
    author="Sam Fok",
    author_email="samfok@stanford.edu",
    packages=find_packages(),
    include_package_data=True,
    scripts=[],
    url="https://github.com/Stanford-BIS/nengo_alif_steady_state.git",
    license="https://github.com/Stanford-BIS/adaptive_lif/blob/master/LICENSE",
    description=description,
    install_requires=[
        "nengo",
    ],
)
