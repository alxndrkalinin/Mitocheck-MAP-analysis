from setuptools import find_packages, setup

setup(
    name="mitocheck-map-analysis",
    version="0.0.1",
    packages=find_packages(),  # Automatically discover and include all packages
    author="Erik Serrano",
    author_email="erik.serrano@cuanschutz.edu",
    description="""Utilizing the mean average precision (MAP) metric to assess
                reproducibility and perturbation effect on single-cell profiles
                in the MitoCheck dataset.""",
    url="https://github.com/WayScience/Mitocheck-MAP-analysis",
)
