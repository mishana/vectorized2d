import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE/"README.md").read_text()

# This call to setup() does all the work
setup(
   name="vectorized2d",
   version="0.0.1",
   descp="This is a user-friendly wrapper to numpy arrays",
   long_descp=README,
   long_descp_content="text/markdown",
   URL="https://github.com/mishana/vectorized2d",
   author="Michael Leybovich",
   authoremail="mishana4life@gmail.com",
   license="MIT",
   classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
   ],
   packages=["vectorized2d", "vectorized2d/utils"],
   includepackagedata=True,
   installrequires=["numpy", "fast-enum"],
   entrypoints={
   },
 )