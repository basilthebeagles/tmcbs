"""Circuit construction: structure, Table III sizes, dispatch, determinism."""
import numpy as np
import pytest
import stim

import tmcbs as t
import tmcbs.experiments as experiments
from third_party.gong_sliding_window_decoder.utils import inverse

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


def _noiseless_block_observables(circ, code, logicals, num_blocks):
    """Return raw logical parities from each final destructively measured block."""
    det = circ.compile_detector_sampler(seed=1).sample(shots=16)
    assert not det.any(), "an injected logical operator produced a syndrome"

    measurements = circ.compile_sampler(seed=1).sample(shots=16)
    final_data = measurements[:, -num_blocks * code.n:].astype(np.uint8)
    final_data = final_data.reshape(16, num_blocks, code.n)
    logicals = np.asarray(logicals, dtype=np.uint8)
    obs = (final_data @ logicals.T) % 2
    assert np.all(obs == obs[0]), "ideal logical output depends on measurement branch"
    return obs[0].astype(np.uint8)


def _noiseless_observables(circ, code):
    """Return logical-Z parities from a circuit with one final output block."""
    return _noiseless_block_observables(circ, code, code.css.lz, num_blocks=1)[0]


def _canonical_logical_x(code):
    """Pair a logical-X basis canonically with the code's stored logical Zs."""
    logical_x = np.asarray(code.css.lx, dtype=np.uint8)
    logical_z = np.asarray(code.css.lz, dtype=np.uint8)
    pairing = (logical_x @ logical_z.T) % 2
    canonical_x = (inverse(pairing) @ logical_x) % 2
    assert np.array_equal((canonical_x @ logical_z.T) % 2,
                          np.eye(code.k, dtype=np.uint8))
    return canonical_x


def _teleportation_with_input_logical_x(code, logical_x, monkeypatch):
    """Build the production teleportation circuit with X_j injected into CB1."""
    real_make_builder = experiments.make_builder

    def make_builder_with_input_x(*args, **kwargs):
        builder = real_make_builder(*args, **kwargs)
        real_init_qubits = builder.initQubits

        def init_qubits_with_input_x(block, *init_args, **init_kwargs):
            real_init_qubits(block, *init_args, **init_kwargs)
            if block == 0:
                builder.logicalOp("X", logical_x, block, errors=False)

        builder.initQubits = init_qubits_with_input_x
        return builder

    # A context keeps the injection local to this one circuit construction.
    with monkeypatch.context() as patch:
        patch.setattr(experiments, "make_builder", make_builder_with_input_x)
        return experiments.teleportation(code, p=0.0, p_ebit=0.0)


@pytest.mark.parametrize(
    "name", ["[[9,1,3]] SC", "[[18,4,4]] BB", "[[36,4,6]] BB"])
def test_teleportation_transfers_each_logical_x_up_to_permutation(name, monkeypatch):
    """Each input logical bit must appear once, possibly relabelled, on CB3.

    Applying logical X_j to the initial |0...0> payload prepares logical bit j in
    |1>.  The final logical-Z measurement vector therefore gives row j of the
    logical input-to-output transfer matrix.  A permutation matrix permits a
    consistent relabelling of logical qubits while rejecting loss, duplication,
    or XOR mixing of their values.
    """
    code = t.get_code(name)
    baseline = _noiseless_observables(t.teleportation(code, 0.0, 0.0), code)
    assert not baseline.any()

    # ``css_code`` returns valid but not necessarily canonically paired logical
    # bases.  Canonical X_j must anticommute with Z_j and commute with every Z_i
    # for i != j; otherwise applying one stored lx row can legitimately flip
    # several stored lz observables before teleportation even begins.
    canonical_x = _canonical_logical_x(code)

    transfer = np.vstack([
        _noiseless_observables(
            _teleportation_with_input_logical_x(code, canonical_x[j], monkeypatch),
            code)
        for j in range(code.k)
    ])

    assert np.all(transfer.sum(axis=1) == 1), \
        f"an input logical X was lost, duplicated, or mixed:\n{transfer}"
    assert np.all(transfer.sum(axis=0) == 1), \
        f"output logical mapping is not one-to-one:\n{transfer}"


def _non_local_cnot_with_input_logical_x(code, block, logical_x, monkeypatch):
    """Build the production non-local CNOT with X_j injected into one input."""
    real_make_builder = experiments.make_builder

    def make_builder_with_input_x(*args, **kwargs):
        builder = real_make_builder(*args, **kwargs)
        real_init_qubits = builder.initQubits

        def init_qubits_with_input_x(init_block, *init_args, **init_kwargs):
            real_init_qubits(init_block, *init_args, **init_kwargs)
            if init_block == block:
                builder.logicalOp("X", logical_x, init_block, errors=False)

        builder.initQubits = init_qubits_with_input_x
        return builder

    with monkeypatch.context() as patch:
        patch.setattr(experiments, "make_builder", make_builder_with_input_x)
        return experiments.non_local_cnot(code, p=0.0, p_ebit=0.0)


@pytest.mark.parametrize(
    "name", ["[[9,1,3]] SC", "[[18,4,4]] BB", "[[36,4,6]] BB"])
def test_non_local_cnot_logical_z_basis_truth_table(name, monkeypatch):
    """The production remote CNOT obeys (c, t) -> (c, c xor t)."""
    code = t.get_code(name)
    logical_x = _canonical_logical_x(code)

    baseline = _noiseless_block_observables(
        t.non_local_cnot(code, 0.0, 0.0), code, code.css.lz, num_blocks=2)
    assert not baseline.any()

    for j in range(code.k):
        from_control = _noiseless_block_observables(
            _non_local_cnot_with_input_logical_x(
                code, block=0, logical_x=logical_x[j], monkeypatch=monkeypatch),
            code, code.css.lz, num_blocks=2)
        expected = np.zeros((2, code.k), dtype=np.uint8)
        expected[:, j] = 1
        assert np.array_equal(from_control, expected), \
            f"logical X_{j} on the control gave\n{from_control}, expected\n{expected}"

        from_target = _noiseless_block_observables(
            _non_local_cnot_with_input_logical_x(
                code, block=1, logical_x=logical_x[j], monkeypatch=monkeypatch),
            code, code.css.lz, num_blocks=2)
        expected = np.zeros((2, code.k), dtype=np.uint8)
        expected[1, j] = 1
        assert np.array_equal(from_target, expected), \
            f"logical X_{j} on the target gave\n{from_target}, expected\n{expected}"


def _data_qubits(builder, code, block):
    """Return a builder block's physical data-qubit indices in CSS column order."""
    cb = builder.cbArr[block]
    if code.is_surface:
        return list(cb[0])
    return list(cb[1]) + list(cb[2])


def _measure_pauli_x(builder, qubits, support):
    """Append an MPP measurement and keep the builder's record offsets valid."""
    targets = []
    for q in np.flatnonzero(support):
        if targets:
            targets.append(stim.target_combiner())
        targets.append(stim.target_x(qubits[q]))
    builder.c.append("MPP", targets)

    # The builders store every referenced measurement as a negative offset from
    # the current record position.  An extra test-only MPP shifts all old records.
    builder.measureTracker[0] += 1
    builder.measureTracker[1] += 1


def _prepare_all_logical_plus(builder, code, block, logical_x):
    """Project an encoded |0...0> block into |+...+>, correcting each sign."""
    data_qubits = _data_qubits(builder, code, block)
    logical_z = np.asarray(code.css.lz, dtype=np.uint8)
    for j in range(code.k):
        _measure_pauli_x(builder, data_qubits, logical_x[j])
        for q in np.flatnonzero(logical_z[j]):
            builder.c.append("CZ", [stim.target_rec(-1), data_qubits[q]])
    builder.c.append("TICK")


def _non_local_cnot_x_basis(code, input_z, monkeypatch):
    """Run the production CNOT on |+> inputs and destructively read logical X."""
    real_make_builder = experiments.make_builder
    logical_x = _canonical_logical_x(code)
    logical_z = np.asarray(code.css.lz, dtype=np.uint8)

    def make_builder_in_x_basis(*args, **kwargs):
        builder = real_make_builder(*args, **kwargs)
        real_transversal_op = builder.transversalOp
        real_measure_data = builder.measureDataQubits
        inputs_prepared = False

        def transversal_op_with_x_inputs(op, blocks, *op_args, **op_kwargs):
            nonlocal inputs_prepared
            type_arr = op_kwargs.get("typeArr", ["BB", "BB"])
            starts_remote_cnot = (
                op == "CX" and list(blocks) == [0, 0]
                and list(type_arr) == ["BB", "e"]
            )
            if starts_remote_cnot and not inputs_prepared:
                _prepare_all_logical_plus(builder, code, 0, logical_x)
                _prepare_all_logical_plus(builder, code, 1, logical_x)
                if input_z is not None:
                    block, j = input_z
                    builder.logicalOp("Z", logical_z[j], block, errors=False)
                inputs_prepared = True
            return real_transversal_op(op, blocks, *op_args, **op_kwargs)

        def measure_data_in_x(blocks, *measure_args, **measure_kwargs):
            measure_kwargs["zBasis"] = False
            return real_measure_data(blocks, *measure_args, **measure_kwargs)

        builder.transversalOp = transversal_op_with_x_inputs
        builder.measureDataQubits = measure_data_in_x
        # The production experiment deliberately declares only final Z-basis
        # detectors/observables.  This phase-side harness reads X directly and
        # omits those final declarations; all round-to-round syndrome detectors
        # from the production circuit remain present and must stay deterministic.
        builder.endOfCircuitDetectorsForLogicalMeasurementReadout = \
            lambda *_args, **_kwargs: None
        builder.obsOffset = lambda *_args, **_kwargs: None
        return builder

    with monkeypatch.context() as patch:
        patch.setattr(experiments, "make_builder", make_builder_in_x_basis)
        return experiments.non_local_cnot(code, p=0.0, p_ebit=0.0)


@pytest.mark.parametrize(
    "name", ["[[9,1,3]] SC", "[[18,4,4]] BB", "[[36,4,6]] BB"])
def test_non_local_cnot_logical_x_basis_truth_table(name, monkeypatch):
    """The phase-side truth table obeys (z_c, z_t) -> (z_c xor z_t, z_t)."""
    code = t.get_code(name)
    logical_x = _canonical_logical_x(code)

    baseline = _noiseless_block_observables(
        _non_local_cnot_x_basis(code, input_z=None, monkeypatch=monkeypatch),
        code, logical_x, num_blocks=2)
    assert not baseline.any()

    for j in range(code.k):
        from_control = _noiseless_block_observables(
            _non_local_cnot_x_basis(code, input_z=(0, j), monkeypatch=monkeypatch),
            code, logical_x, num_blocks=2)
        expected = np.zeros((2, code.k), dtype=np.uint8)
        expected[0, j] = 1
        assert np.array_equal(from_control, expected), \
            f"logical Z_{j} on the control gave\n{from_control}, expected\n{expected}"

        from_target = _noiseless_block_observables(
            _non_local_cnot_x_basis(code, input_z=(1, j), monkeypatch=monkeypatch),
            code, logical_x, num_blocks=2)
        expected = np.zeros((2, code.k), dtype=np.uint8)
        expected[:, j] = 1
        assert np.array_equal(from_target, expected), \
            f"logical Z_{j} on the target gave\n{from_target}, expected\n{expected}"
