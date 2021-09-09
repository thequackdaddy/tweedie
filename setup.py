#!/usr/bin/env python
from setuptools import setup, find_packages

import versioneer


def read(filename):
    with open(filename, 'r') as f:
        return f.read()


setup(name='tweedie',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Tweedie distribution estimations of pdf and cdf',
      url='https://github.com/thequackdaddy/tweedie',
      author='Peter Quackenbush',
      author_email='pquack@gmail.com',
      license='BSD',
      keywords='statsmodels scipy distribution tweedie',
      packages=find_packages(),
      install_requires=['scipy', 'numpy'],
      extras_require={'dev': ['pytest', 'tox']},
      long_description=read('README.rst'),
      zip_safe=False,
      )
