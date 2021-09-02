tweedie is a Python library implementing scipy's ``rv_continuous`` class
for the Tweedie family. The Tweedie family is a member of the exponential
dispersion model family and is commonly used in the insurance indsutry
to model claim amounts for insurance policies (exposure).

The main focus of this package is the compound-Poisson behavior,
specifically where :math:`1 < p < 2`. However, it should be possible to
calculate the distribution for all the possible values of p.

.. image:: https://travis-ci.org/thequackdaddy/tweedie.png?branch=master
   :target: https://travis-ci.org/thequackdaddy/tweedie

.. image:: http://codecov.io/github/thequackdaddy/tweedie/coverage.svg?branch=master
   :target: http://codecov.io/github/thequackdaddy/tweedie?branch=master

Documentation:
  Some of the functions are well documented. Others, not so well. In future
  versions, I might try to publish.

Downloads:
  https://github.com/thequackdaddy/tweedie

Dependencies:
  * Python (2.7, or 3.4+)
  * numpy
  * scipy

Optional dependencies:
  * pytest: needed to run tests
  * tox: needed to run tests against multiple python/scipy versions

Install:
  To run tests against multiple python/scipy versions, install the package using ``pip install -e .[dev]``.

Running the tests:

To test against the current python version and scipy `pytest ./tweedie`

To test against multiple python versions and scipy versions `tox -p all`.
If you run in to issues with performance, just use `tox`.

Code and bug tracker:
  https://github.com/thequackdaddy/tweedie

License:
  2-clause BSD, see LICENSE.txt for details.
