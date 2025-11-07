from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = "0.0.3"
DESCRIPTION = "Utilities for detecting and segmenting movement events (trips or operational windows) from GPS-like latitude/longitude time series."
LONG_DESCRIPTION = ""

# Setting up
setup(
    name="trips_preprocessing",
    version=VERSION,
    author=["Ian dos Anjos Melo Aguiar"],
    author_email="<iannaianjos@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=["pandas", "numpy", "numba"],
    keywords=["python", "regression"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
