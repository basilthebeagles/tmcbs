"""Circuit construction: structure, Table III sizes, dispatch, determinism."""
import pytest

import tmcbs as t

P, P_EBIT = 1e-3, 1e-3

# (rows, cols) = (detectors, distinct error mechanisms) of the space-time PCM,
# from Table III of the paper. A strong end-to-end check on the constructed
# circuits and their detector error models.
TABLE_III_CNOT = {
    "[[9,1,3]] SC": (64, 245),
    "[[25,1,5]] SC": (192, 847),
    "[[18,4,4]] BB": (144, 1314),
    "[[36,4,6]] BB": (288, 2628),
}
TABLE_III_TELE = {
    "[[9,1,3]] SC": (64, 238),
    "[[25,1,5]] SC": (192, 832),
    "[[18,4,4]] BB": (144, 1314),
    "[[36,4,6]] BB": (288, 2646),
}


@pytest.mark.parametrize("name,rc", TABLE_III_CNOT.items())
def test_non_local_cnot_pcm_size_matches_table3(name, rc):
    circ = t.non_local_cnot(t.get_code(name), P, P_EBIT)
    chk, obs, priors = t.parity_check_matrices(circ)
    assert chk.shape == rc
    assert chk.shape[0] == circ.num_detectors
    assert chk.shape[1] == obs.shape[1] == len(priors)


@pytest.mark.parametrize("name,rc", TABLE_III_TELE.items())
def test_teleportation_pcm_size_matches_table3(name, rc):
    circ = t.teleportation(t.get_code(name), P, P_EBIT)
    chk, _, _ = t.parity_check_matrices(circ)
    assert chk.shape == rc


@pytest.mark.parametrize("name", ["[[9,1,3]] SC", "[[18,4,4]] BB"])
def test_observable_counts(name):
    code = t.get_code(name)
    # non-local CNOT exposes 2k observables, teleportation and memory expose k.
    assert t.non_local_cnot(code, P, P_EBIT).num_observables == 2 * code.k
    assert t.teleportation(code, P, P_EBIT).num_observables == code.k
    assert t.memory(code, P).num_observables == code.k


def test_construction_is_deterministic():
    code = t.surface_code(3)
    a = t.non_local_cnot(code, P, P_EBIT)
    b = t.non_local_cnot(code, P, P_EBIT)
    assert str(a) == str(b)


@pytest.mark.parametrize("name", ["[[9,1,3]] SC", "[[18,4,4]] BB"])
def test_build_for_experiment_matches_direct_calls(name):
    code = t.get_code(name)
    E = t.Experiment
    assert str(t.build_for_experiment(code, E.MEMORY, P)) == str(t.memory(code, P))
    assert str(t.build_for_experiment(code, E.NON_LOCAL_CNOT, P, P_EBIT)) \
        == str(t.non_local_cnot(code, P, P_EBIT))
    assert str(t.build_for_experiment(code, E.TELEPORTATION, P, P_EBIT)) \
        == str(t.teleportation(code, P, P_EBIT))
    assert str(t.build_for_experiment(code, E.PAULI_PRODUCT_MEASUREMENT, P, P_EBIT,
                                      ppm_pattern="101")) \
        == str(t.pauli_product_measurement(code, P, P_EBIT, "101"))


def test_ppm_pattern_string_and_list_agree():
    code = t.surface_code(3)
    assert str(t.pauli_product_measurement(code, P, P_EBIT, "101")) \
        == str(t.pauli_product_measurement(code, P, P_EBIT, [1, 0, 1]))


def test_ppm_requires_pattern():
    with pytest.raises(ValueError):
        t.build_for_experiment(t.surface_code(3), t.Experiment.PAULI_PRODUCT_MEASUREMENT,
                               P, P_EBIT)


def test_decoherence_changes_the_circuit():
    code = t.surface_code(3)
    plain = t.non_local_cnot(code, P, P_EBIT)
    decohered = t.non_local_cnot(code, P, P_EBIT, decoherence_ratio=1e-2)
    assert str(plain) != str(decohered)
