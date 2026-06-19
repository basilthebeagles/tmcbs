"""Decoding: PCM extraction and logical-error counting."""
import numpy as np
import pytest

import tmcbs as t

# BP-OSD (from ldpc) is a hard dependency, so use it for portable tests.
DECODER = "BP-OSD"


def test_parity_check_matrices_shapes():
    circ = t.non_local_cnot(t.surface_code(3), 1e-3, 1e-3)
    chk, obs, priors = t.parity_check_matrices(circ)
    assert chk.shape[0] == circ.num_detectors
    assert obs.shape[0] == circ.num_observables
    assert chk.shape[1] == obs.shape[1] == len(priors)
    assert np.all((priors >= 0) & (priors <= 1))


def test_count_logical_errors_in_range():
    circ = t.memory(t.surface_code(3), 1e-2)
    n = t.count_logical_errors(circ, 200, p=1e-2, decoder=DECODER)
    assert isinstance(n, int)
    assert 0 <= n <= 200


def test_count_logical_errors_zero_shots():
    circ = t.memory(t.surface_code(3), 1e-2)
    assert t.count_logical_errors(circ, 0, decoder=DECODER) == 0


def test_unknown_decoder_raises():
    circ = t.memory(t.surface_code(3), 1e-2)
    with pytest.raises(ValueError):
        t.count_logical_errors(circ, 10, decoder="not-a-decoder")


@pytest.mark.parametrize("decoder", t.DECODERS)
def test_each_decoder_runs_or_is_unavailable(decoder):
    circ = t.non_local_cnot(t.surface_code(3), 5e-3, 5e-3)
    try:
        n = t.count_logical_errors(circ, 100, p=5e-3, decoder=decoder)
    except RuntimeError as exc:           # backend not importable on this platform
        pytest.skip(str(exc))
    assert 0 <= n <= 100
