"""LER estimation and the Bayes-factor confidence interval."""
import math

import pytest

import tmcbs as t


def test_bayes_interval_basic():
    lo, hi = t.bayes_factor_interval(5, 1000)
    assert 0 < lo < 0.005 < hi < 1


def test_bayes_interval_zero_errors():
    lo, hi = t.bayes_factor_interval(0, 1000)
    assert lo == 0.0 and 0 < hi < 1


def test_bayes_interval_all_errors():
    lo, hi = t.bayes_factor_interval(1000, 1000)
    assert hi == 1.0 and 0 < lo < 1


@pytest.mark.parametrize("x,n", [(5, 1000), (1, 100000), (250, 1000)])
def test_bayes_interval_matches_sinter(x, n):
    sinter = pytest.importorskip("sinter")
    r = sinter.fit_binomial(num_shots=n, num_hits=x, max_likelihood_factor=1000)
    lo, hi = t.bayes_factor_interval(x, n)
    assert math.isclose(lo, r.low, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(hi, r.high, rel_tol=1e-6, abs_tol=1e-12)


def test_estimate_ler_result_is_consistent():
    code = t.surface_code(3)
    circ = t.non_local_cnot(code, 1e-2, 1e-2)
    res = t.estimate_ler(circ, p=1e-2, d=code.d, decoder="BP-OSD",
                         target_errors=5, initial_batch=200)
    assert isinstance(res, t.LERResult)
    assert res.shots > 0
    assert res.ler == res.errors / res.shots
    assert res.ci_low <= res.ler <= res.ci_high
    assert res.decoder == "BP-OSD"
    # stopped either on the error target or the shot cap
    assert res.errors >= 5 or res.hit_shot_limit
