from setuptools import find_packages, setup

setup(
    name="INV2A",
    version="0.0.1",
    packages=find_packages(),
    install_requires=open("requirements.txt").readlines()
)