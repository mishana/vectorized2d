import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE/"README.md").read_text()

# This call to setup() does all the work
setup(
   name="vectorized2d",
   version="0.0.5",
   description="This is a user-friendly wrapper to numpy arrays, \
   with batteries included and numba-enhanced performance.",
   long_description=README,
   long_description_content_type="text/markdown",
   url="https://github.com/mishana/vectorized2d",
   author="Michael Leybovich",
   author_email="mishana4life@gmail.com",
   license="MIT",
   classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
   ],
   packages=["vectorized2d", "vectorized2d/utils"],
   include_package_data=True,
   install_requires=["numpy", "fast-enum"],
   entrypoints={
   },
 )