tweedie is a Python library implementing scipy's ``rv_continuous`` class
for the Tweedie family. The Tweedie family is a member of the exponential
dispersion model family and is commonly used in the insurance indsutry
to model losses on policies.

The main focus of this package is the compound-Poisson behavior,
specifically where :math:`1 < p < 2`. However, it should theoretically
be possible to all the possible values of p.

.. image:: https://travis-ci.org/thequackdaddy/tweedie.png?branch=master
   :target: https://travis-ci.org/thequackdaddy/tweedie

Documentation:
  Some of the functions are well documents. Others, not so well. In future
  versions, I might try to publish.

Downloads:
  https://github.com/thequackdaddy/tweedie

Dependencies:
  * Python (2.7, or 3.4+)
  * numpy
  * scipy

Optional dependencies:
  * pytest: needed to run tests

Install:
  At the moment, ``python setup.py install`` will work. I'll try to put this on
  pypi at some point in the future.

Code and bug tracker:
  https://github.com/thequackdaddy/tweedie

License:
  2-clause BSD, see LICENSE.txt for details.
