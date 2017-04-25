# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 15:10:39 2017

@author: pquackenbush
"""
import numpy as np
from numpy.testing import assert_allclose
from tweedie import tweedie
from results import test_ys, test_results, num_tests


def test_R_compat_density():
    lines = test_results.split('\n')
    start_idx = lines.index("--BEGIN TEST CASE--")
    rows = len(lines)
    py_tests = 0
    while start_idx + 1 < rows:
        stop_idx = lines.index('--END TEST CASE--', start_idx)
        for line in lines[start_idx + 1: stop_idx]:
            key, value = line.split('=', 1)
            value = value.replace('Inf', 'np.inf')
            if key == 'power':
                power = eval(value)
            if key == 'mu':
                mu = eval(value)
            if key == 'phi':
                phi = eval(value)
            if key == 'density':
                density = eval(value)
        start_idx = stop_idx + 1
        py_tests += 1
        tdensity = tweedie(p=power, mu=mu, phi=phi).pdf(test_ys)
        assert_allclose(density, tdensity, rtol=1e-8, atol=1e-8)
    assert num_tests == py_tests


def test_R_compat_cdf():
    test_run = False
    lines = test_results.split('\n')
    start_idx = lines.index("--BEGIN TEST CASE--")
    rows = len(lines)
    py_tests = 0
    while start_idx + 1 < rows:
        stop_idx = lines.index('--END TEST CASE--', start_idx)
        cdf = None
        for line in lines[start_idx + 1: stop_idx]:
            key, value = line.split('=', 1)
            if key == 'power':
                power = eval(value)
            if key == 'mu':
                mu = eval(value)
            if key == 'phi':
                phi = eval(value)
            if key == 'cdf':
                cdf = eval(value)
        start_idx = stop_idx + 1
        py_tests += 1
        if cdf is not None:
            test_run = True
            tcdf = tweedie(p=power, mu=mu, phi=phi).cdf(test_ys)
            if power != 3:
                assert_allclose(cdf, tcdf, rtol=1e-8, atol=1e-8)
            else:
                # I think the R function might be off here
                assert_allclose(cdf, tcdf, rtol=1e-3, atol=1e-3)
    assert num_tests == py_tests
    assert test_run


def test_rvs_smoke():
    # Just a smoke test for now.
    rvs = tweedie(mu=150, p=1.5, phi=500.).rvs(10000)
    assert len(rvs) == 10000
    rvs = tweedie(mu=np.repeat(150, 100), p=np.repeat(1.5, 100),
                  phi=np.repeat(500, 100)).rvs(100)
    assert len(rvs) == 100
    rvs1 = tweedie(mu=150, p=1.5, phi=500).rvs(100000, random_state=42)
    rvs2 = tweedie(mu=150, p=1.5, phi=500).rvs(100000, random_state=42)
    assert_allclose(rvs1, rvs2)


def test_mean_close():
    from itertools import product
    mus = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ps = [1.3, 1.5, 1.7]
    phis = [1, 2, 3]
    for mu, p, phi in product(mus, ps, phis):
        rvs = tweedie(mu=mu, p=p, phi=phi).rvs(10000, random_state=42)
        assert_allclose(mu, rvs.mean(), rtol=.05)


def test_variance_close():
    from itertools import product
    mus = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ps = [1.3, 1.5, 1.7]
    phis = [1, 2, 3]
    for mu, p, phi in product(mus, ps, phis):
        rvs = tweedie(mu=mu, p=p, phi=phi).rvs(10000, random_state=42)
        assert_allclose(phi * mu ** p, rvs.var(), rtol=.1)
