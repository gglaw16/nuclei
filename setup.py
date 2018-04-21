#!/usr/bin/env python

from setuptools import setup, find_packages
from os import path
here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
        long_description = f.read()

setup(name='nuclei',
      version='0.1',
      description='Maniulate czi files in girder',
      long_description=long_description,
      #long_description_content_type='text/markdown',
      author = 'Gwenda Law',
      author_email='gwenda.law@gmail.com',
      #packages=['girder', 'girder.keys', 'deep_learning'],
      packages=find_packages(exclude=['tests']),
      install_requires=['girder_client', 'numpy', 'opencv-python', 'scipy', 'urllib3'],
)


