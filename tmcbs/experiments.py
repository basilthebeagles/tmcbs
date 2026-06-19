"""Circuit constructors for the distributed transversal primitives.

Each function here assembles a complete, noisy ``stim.Circuit`` for one logical
operation by driving a **circuit builder** -- ``CircuitBuilder`` for bivariate
bicycle codes or ``CircuitBuilderSurface`` for rotated surface codes. The two
builders share a method vocabulary, so every constructor below is written once and
works for both code families; ``Code.is_surface`` selects the right builder.

The builder vocabulary (see ``generalCircuitBuilder*.py``):

``initQubits(block, errors=True)``
    Noisy reset of one logical code block.
``prepareEbits(transError)``
    Create ``n`` Bell pairs across the link; ``transError`` is the ebit infidelity
    (2-qubit depolarising applied after a perfect Bell preparation).
``includeDelayDueToEbitGenTime(ratio, blocks, types, perCycle)``
    Apply T1/T2 decoherence to ebits according to how long they waited to be used.
``extractionRound(blocks, firstPass=..., errors=..., zBasis=...)``
    One syndrome-extraction cycle over the given blocks (declares detectors;
    ``firstPass`` opens detectors, later rounds compare to the previous round).
``transversalOp(op, [a, b], typeArr=[...])``
    Transversal 1- or 2-qubit gate. ``typeArr`` entries are ``"BB"`` (a logical
    code block) or ``"e"`` (an ebit register), e.g. a transversal CNOT from a code
    block onto an ebit is ``transversalOp("CX", [block, 0], typeArr=["BB", "e"])``.
``measureEbit0ThenCorrectEbit1()`` / ``measureEbit1ThenCorrectCB(block)``
    The LOCC half of a non-local CNOT: measure one ebit register and apply the
    classically-conditioned Pauli correction to the other register / a code block.
``measureCBThenCorrectCB(op, [a, b])``
    Measure code block ``a`` and conditionally apply ``op`` to block ``b`` (the
    feed-forward used in teleportation).
``measureDataQubits(blocks)``
    Destructive logical readout (Z basis) of the data qubits.
``endOfCircuitDetectorsForLogicalMeasurementReadout(block)`` and
``obsOffset(block, offset)``
    Reconstruct the final stabilisers as detectors from the data-qubit readout, and
    declare the logical observables (one per logical qubit, placed at ``offset``).

Each constructor maps one-to-one onto the builder calls for its protocol, so the
circuits stay easy to read and to fork into your own gadgets (see notebook 04).

Experiment indices (kept compatible with ``runEbitExperimentMPI.py --experiment``):

=  =========================  ============================
0  MEMORY                     :func:`memory`
1  NON_LOCAL_CNOT             :func:`non_local_cnot`
2  TELEPORTATION              :func:`teleportation`
3  LOCAL_CNOT                 :func:`local_cnot`
4  LOCAL_TELEPORTATION        :func:`local_teleportation`
5  PAULI_PRODUCT_MEASUREMENT  :func:`pauli_product_measurement`
=  =========================  ============================
"""

from __future__ import annotations

from enum import IntEnum
from typing import List, Sequence, Union

import stim

from . import generalCircuitBuilder
from . import generalCircuitBuilderSurface
from .codes import Code

# Number of syndrome-extraction rounds inserted between operations for the
# ebit-based primitives (the surface/BB memory uses ``d`` rounds instead). These
# match the values in the paper Methods and the original circuit definitions.
_ROUNDS_BETWEEN_OPS = 3


class Experiment(IntEnum):
    """Stable integer ids for each primitive (match the CLI ``--experiment``)."""

    MEMORY = 0
    NON_LOCAL_CNOT = 1
    TELEPORTATION = 2
    LOCAL_CNOT = 3
    LOCAL_TELEPORTATION = 4
    PAULI_PRODUCT_MEASUREMENT = 5


# --------------------------------------------------------------------------- #
# Builder construction.
# --------------------------------------------------------------------------- #
def make_builder(code: Code, num_blocks: int, p: float, *, ebits: bool = False):
    """Create the right circuit builder for ``code`` with ``num_blocks`` blocks.

    Returns a ``CircuitBuilder`` (BB codes) or ``CircuitBuilderSurface`` (surface
    codes) configured with physical error rate ``p`` and, if ``ebits=True``, an
    extra pair of ebit registers for cross-link Bell pairs. Use this to compose
    your own circuits from the builder vocabulary (see notebook 04).
    """
    if code.is_surface:
        return generalCircuitBuilderSurface.CircuitBuilderSurface(
            code.builder_object, code.d, p, code.n, num_blocks, ebits=ebits)
    return generalCircuitBuilder.CircuitBuilder(
        code.builder_object, code.d, p, code.n, num_blocks, ebits=ebits)


# --------------------------------------------------------------------------- #
# Primitives.
# --------------------------------------------------------------------------- #
def memory(code: Code, p: float) -> stim.Circuit:
    """Single-block memory experiment: init, ``d`` syndrome rounds, readout.

    The baseline fault-tolerance check: a logical qubit is initialised, held for
    ``d`` rounds of syndrome extraction, then measured. With a good code and
    decoder the logical error rate falls as ``p`` decreases.
    """
    b = make_builder(code, 1, p, ebits=False)
    b.initQubits(0, errors=True)
    b.extractionRound([0], firstPass=True, errors=True)
    for _ in range(code.d - 1):
        b.extractionRound([0], firstPass=False, errors=True)
    b.measureDataQubits([0])
    b.endOfCircuitDetectorsForLogicalMeasurementReadout(0)
    b.obsOffset(0, 0)
    return b.getCirc()


def non_local_cnot(
    code: Code,
    p: float,
    p_ebit: float,
    *,
    decoherence_ratio: float = 0.0,
    ebits_per_cycle: int = -1,
) -> stim.Circuit:
    """Transversal non-local CNOT between two blocks on different nodes (Fig. 1b).

    Block 0 (on QC1) is the control, block 1 (on QC2) the target. The operation
    consumes ``n`` ebits and uses only transversal gates, measurements and
    classical feed-forward:

    1. initialise both code blocks and prepare ``n`` noisy ebits across the link;
    2. ``d``-like syndrome extraction to settle the encoded state
       (``_ROUNDS_BETWEEN_OPS`` rounds here);
    3. transversal CNOT (control block 0 -> ebit register 0), then measure ebit
       register 0 and feed forward an X correction onto ebit register 1;
    4. transversal CNOT (ebit register 1 -> target block 1), a transversal H on the
       ebits, measure ebit register 1 and feed forward a Z correction onto block 0;
    5. more syndrome extraction, then destructive readout of both blocks.

    Parameters
    ----------
    p : physical error rate.
    p_ebit : ebit infidelity.
    decoherence_ratio : oldest-ebit age / T1 (0 disables ebit delay noise; this is
        the x-axis of the decoherence study, Fig. 4).
    ebits_per_cycle : ebit production schedule. ``-1`` = one-at-a-time; a positive
        value = ``O(d)``-style "line" generation, reduced to the nearest divisor of
        ``n`` (pass ``code.d`` for the line-generation curves of Fig. 4).
    """
    t1t2 = _is_decohering(decoherence_ratio)
    per_cycle = ebits_per_cycle
    if t1t2 and per_cycle != -1:
        per_cycle = _divisor_at_most(code.n, per_cycle)

    b = make_builder(code, 2, p, ebits=True)
    b.initQubits(0)
    b.initQubits(1)
    b.prepareEbits(transError=p_ebit)
    if t1t2:
        b.includeDelayDueToEbitGenTime(decoherence_ratio, [0, 1], ["e", "e"], per_cycle)
    b.extractionRound([0, 1], firstPass=True)
    for _ in range(_ROUNDS_BETWEEN_OPS):
        b.extractionRound([0, 1], firstPass=False, errors=True)

    b.transversalOp("CX", [0, 0], typeArr=["BB", "e"])
    b.measureEbit0ThenCorrectEbit1()
    b.transversalOp("CX", [1, 1], typeArr=["e", "BB"])
    b.transversalOp("H", [1], typeArr=["e"])
    b.measureEbit1ThenCorrectCB(0)

    for _ in range(_ROUNDS_BETWEEN_OPS):
        b.extractionRound([0, 1], firstPass=False, errors=True)
    b.measureDataQubits([0, 1])
    b.endOfCircuitDetectorsForLogicalMeasurementReadout(0)
    b.endOfCircuitDetectorsForLogicalMeasurementReadout(1)
    b.obsOffset(0, 0)
    b.obsOffset(1, code.k)
    return b.getCirc()


def teleportation(
    code: Code,
    p: float,
    p_ebit: float,
    *,
    decoherence_ratio: float = 0.0,
) -> stim.Circuit:
    """Full logical teleportation of a block from QC1 to QC2 (Fig. 1c).

    Blocks: 0 = the state |psi> to teleport (QC1), 1 = a |0> ancilla (QC1), 2 = a
    |0> ancilla (QC2). A logical Bell pair is built between blocks 1 and 2 via a
    non-local CNOT, then a logical Bell-state measurement on blocks 0 and 1 (with
    feed-forward corrections to block 2) completes the teleport. Block 2 carries
    |psi> at the end. This uses three code blocks and more feed-forward than the
    non-local CNOT, which is why its logical error rate is higher at matched
    parameters.
    """
    t1t2 = _is_decohering(decoherence_ratio)

    b = make_builder(code, 3, p, ebits=True)
    b.initQubits(0, errors=True)
    b.initQubits(1)
    b.initQubits(2)
    b.prepareEbits(transError=p_ebit)
    if t1t2:
        b.includeDelayDueToEbitGenTime(decoherence_ratio, [0, 1], ["e", "e"])

    # Settle block 0 (the payload) and the Bell-pair blocks 1, 2.
    b.extractionRound([0], firstPass=True, errors=True)
    for _ in range(_ROUNDS_BETWEEN_OPS):
        b.extractionRound([0], firstPass=False, errors=True)
    b.extractionRound([1, 2], firstPass=True)
    for _ in range(_ROUNDS_BETWEEN_OPS):
        b.extractionRound([1, 2], firstPass=False, errors=True)

    # Logical Bell pair between blocks 1 and 2 (transversal H + non-local CNOT).
    b.transversalOp("H", [1], typeArr=["BB"])
    b.transversalOp("CX", [1, 0], typeArr=["BB", "e"])
    b.measureEbit0ThenCorrectEbit1()
    b.transversalOp("CX", [1, 2], typeArr=["e", "BB"])
    b.transversalOp("H", [1], typeArr=["e"])
    b.measureEbit1ThenCorrectCB(1)

    # Logical Bell-state measurement on blocks 0 and 1, feeding forward to block 2.
    b.transversalOp("CX", [0, 1], typeArr=["BB", "BB"])
    b.transversalOp("H", [0], typeArr=["BB"])
    b.measureCBThenCorrectCB("CX", [1, 2])
    b.measureCBThenCorrectCB("CZ", [0, 2])

    for _ in range(_ROUNDS_BETWEEN_OPS):
        b.extractionRound([2], firstPass=False, errors=True)
    b.measureDataQubits([2])
    b.endOfCircuitDetectorsForLogicalMeasurementReadout(2)
    b.obsOffset(2, 0)
    return b.getCirc()


def local_cnot(
    code: Code, p: float, p_ebit: float = 0.0, *, custom_noise: float = -1,
) -> stim.Circuit:
    """Same-device transversal CNOT between two blocks -- a no-ebit baseline.

    Round-matched to :func:`non_local_cnot`: ``_ROUNDS_BETWEEN_OPS`` settling rounds
    before and after the gate (rather than ``d - 1``), so the local and non-local
    CNOTs are compared over the same number of syndrome-extraction rounds.

    Parameters
    ----------
    p_ebit : accepted for a uniform signature but unused (there are no ebits).
    custom_noise : two-qubit depolarising rate applied after the transversal CNOT.
        ``-1`` (the default) uses the builder's standard post-Clifford rate; set a
        value to model an extra-noisy transversal gate -- the local analogue of the
        non-local CNOT's ``p_ebit``.
    """
    b = make_builder(code, 2, p, ebits=False)
    b.initQubits(0)
    b.initQubits(1)
    b.extractionRound([0, 1], firstPass=True)
    for _ in range(_ROUNDS_BETWEEN_OPS):
        b.extractionRound([0, 1], firstPass=False, errors=True)
    b.transversalOp("CX", [0, 1], typeArr=["BB", "BB"], customNoise=custom_noise)
    for _ in range(_ROUNDS_BETWEEN_OPS):
        b.extractionRound([0, 1], firstPass=False, errors=True)
    b.measureDataQubits([0, 1])
    b.endOfCircuitDetectorsForLogicalMeasurementReadout(0)
    b.endOfCircuitDetectorsForLogicalMeasurementReadout(1)
    b.obsOffset(0, 0)
    b.obsOffset(1, code.k)
    return b.getCirc()


def local_teleportation(
    code: Code, p: float, p_ebit: float = 0.0, *, custom_noise: float = -1,
) -> stim.Circuit:
    """Same-device teleportation -- a no-ebit baseline for :func:`teleportation`.

    Round-matched to :func:`teleportation`: the local Bell pair (a single transversal
    CNOT) stands in for the non-local CNOT, and the circuit uses the same settling
    cadence and the same total number of syndrome-extraction rounds (no extra
    settling between the Bell pair and the Bell-state measurement, and
    ``_ROUNDS_BETWEEN_OPS`` rounds before the final readout).

    Parameters
    ----------
    p_ebit : accepted for a uniform signature but unused (there are no ebits).
    custom_noise : two-qubit depolarising rate applied after the Bell-pair transversal
        CNOT -- the local analogue of the non-local CNOT carrying ``p_ebit``. ``-1``
        (the default) uses the builder's standard post-Clifford rate.
    """
    b = make_builder(code, 3, p, ebits=False)
    b.initQubits(0, errors=True)
    b.initQubits(1)
    b.initQubits(2)
    b.extractionRound([0], firstPass=True, errors=True)
    for _ in range(_ROUNDS_BETWEEN_OPS):
        b.extractionRound([0], firstPass=False, errors=True)
    b.extractionRound([1, 2], firstPass=True)
    for _ in range(_ROUNDS_BETWEEN_OPS):
        b.extractionRound([1, 2], firstPass=False, errors=True)
    b.transversalOp("H", [1], typeArr=["BB"])
    b.transversalOp("CX", [1, 2], typeArr=["BB", "BB"], customNoise=custom_noise)
    b.transversalOp("CX", [0, 1], typeArr=["BB", "BB"])
    b.transversalOp("H", [0], typeArr=["BB"])
    b.measureCBThenCorrectCB("CX", [1, 2])
    b.measureCBThenCorrectCB("CZ", [0, 2])
    for _ in range(_ROUNDS_BETWEEN_OPS):
        b.extractionRound([2], firstPass=False, errors=True, zBasis=False)
    b.measureDataQubits([2])
    b.endOfCircuitDetectorsForLogicalMeasurementReadout(2)
    b.obsOffset(2, 0)
    return b.getCirc()


def pauli_product_measurement(
    code: Code,
    p: float,
    p_ebit: float,
    pattern: Union[str, Sequence[int]],
) -> stim.Circuit:
    """Chained Pauli-string compute step via transversal CNOTs (Fig. 5a).

    Computes the parity of ``w`` data blocks onto an ancilla block with a chain of
    CNOTs ``CX(0->1), CX(1->2), ..., CX(w-1 -> ancilla)`` (paper Methods, "Pauli
    string compute"). ``pattern`` is the per-link sequence: ``0`` = local CNOT,
    ``1`` = non-local CNOT. A pattern of length ``L`` realises a weight ``w = L + 1``
    Pauli string (plus the ancilla). Examples: ``"1"`` (w=2), ``"10"`` (w=3),
    ``"101"`` (w=4), ``"10101"`` (w=6, the d=8 run in the paper).
    """
    instructions = _parse_pattern(pattern)
    num_blocks = len(instructions) + 2  # w data/ancilla blocks (= L+1) plus ancilla

    b = make_builder(code, num_blocks, p, ebits=True)
    blocks = list(range(num_blocks))
    for i in blocks:
        b.initQubits(i)
    b.extractionRound(blocks, firstPass=True)
    for _ in range(_ROUNDS_BETWEEN_OPS):
        b.extractionRound(blocks, firstPass=False, errors=True)

    # Walk the chain: link i couples block i (control) to block i+1 (target).
    for i, instruction in enumerate(instructions):
        control, target = blocks[i], blocks[i + 1]
        if int(instruction) == 0:                      # local transversal CNOT
            b.transversalOp("CX", [control, target], typeArr=["BB", "BB"])
        elif int(instruction) == 1:                    # non-local CNOT via ebits
            b.prepareEbits(transError=p_ebit)
            b.transversalOp("CX", [control, 0], typeArr=["BB", "e"])
            b.measureEbit0ThenCorrectEbit1()
            b.transversalOp("CX", [1, target], typeArr=["e", "BB"])
            b.transversalOp("H", [1], typeArr=["e"])
            b.measureEbit1ThenCorrectCB(control)
        b.extractionRound([control, target], firstPass=False, errors=True)

    # Final local CNOT onto the ancilla block, then read everything out.
    b.transversalOp("CX", [blocks[-2], blocks[-1]], typeArr=["BB", "BB"])
    b.extractionRound([blocks[-2], blocks[-1]], firstPass=False, errors=True)
    for count, block in enumerate(blocks):
        b.measureDataQubits([block])
        b.endOfCircuitDetectorsForLogicalMeasurementReadout(block)
        b.obsOffset(block, code.k * count)
    return b.getCirc()


# --------------------------------------------------------------------------- #
# Dispatch + helpers.
# --------------------------------------------------------------------------- #
def build_for_experiment(
    code: Code,
    experiment: Union[Experiment, int],
    p: float,
    p_ebit: float = 0.0,
    *,
    decoherence_ratio: float = 0.0,
    ebits_per_cycle: int = -1,
    ppm_pattern: Union[str, Sequence[int], None] = None,
    custom_noise: float = -1,
) -> stim.Circuit:
    """Dispatch on an :class:`Experiment` id and return the assembled circuit.

    The single entry point shared by the notebooks and the MPI runner, so the
    mapping from experiment id to constructor lives in exactly one place.
    """
    experiment = Experiment(int(experiment))
    if experiment is Experiment.MEMORY:
        return memory(code, p)
    if experiment is Experiment.NON_LOCAL_CNOT:
        return non_local_cnot(code, p, p_ebit, decoherence_ratio=decoherence_ratio,
                              ebits_per_cycle=ebits_per_cycle)
    if experiment is Experiment.TELEPORTATION:
        return teleportation(code, p, p_ebit, decoherence_ratio=decoherence_ratio)
    if experiment is Experiment.LOCAL_CNOT:
        return local_cnot(code, p, p_ebit, custom_noise=custom_noise)
    if experiment is Experiment.LOCAL_TELEPORTATION:
        return local_teleportation(code, p, p_ebit, custom_noise=custom_noise)
    if experiment is Experiment.PAULI_PRODUCT_MEASUREMENT:
        if ppm_pattern is None:
            raise ValueError("PAULI_PRODUCT_MEASUREMENT requires ppm_pattern, "
                             "e.g. ppm_pattern='101'.")
        return pauli_product_measurement(code, p, p_ebit, ppm_pattern)
    raise ValueError(f"Unhandled experiment {experiment!r}.")


def _is_decohering(decoherence_ratio: float) -> bool:
    return decoherence_ratio not in (0, 0.0, None)


def _divisor_at_most(n: int, target: int) -> int:
    """Largest divisor of ``n`` that is ``<= target`` (mirrors the MPI runner)."""
    f = target
    while f > 1 and n % f != 0:
        f -= 1
    return max(f, 1)


def _parse_pattern(pattern: Union[str, Sequence[int]]) -> List[int]:
    if isinstance(pattern, str):
        cleaned = pattern.replace(",", "").strip()
        return [int(ch) for ch in cleaned]
    return [int(x) for x in pattern]
