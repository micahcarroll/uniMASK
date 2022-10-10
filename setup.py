#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="uniMASK",
    version="0.0.1",
    description="Flexible inference and generation of human-like behavior.",
    long_description=long_description,
    author="Micah Carroll",
    author_email="mdc@berkeley.edu",
    packages=find_packages(),
    install_requires=["gym_minigrid", "mujoco-py<2.2,>=2.1"],
)
