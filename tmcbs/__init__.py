"""TMCBS -- Transversal Multiple Code Block Simulator.

Circuit-level simulation of distributed transversal fault-tolerant operations
(non-local CNOT, logical teleportation, Pauli-product-measurement compute steps)
between rotated surface codes and bivariate-bicycle qLDPC codes, using explicit
noisy ebits.

Typical use::

    import tmcbs as t

    code = t.surface_code(3)                       # or t.bb_code("[[18,4,4]] BB")
    circ = t.non_local_cnot(code, p=1e-3, p_ebit=1e-3)
    res  = t.estimate_ler(circ, p=1e-3, d=code.d, decoder="teser",
                          target_errors=50)
    print(res)                                     # LER + Bayes-factor error bars

The low-level syndrome-extraction circuit construction lives in
``general_circuit_builder.py`` / ``general_circuit_builder_surface.py``; this
package assembles the distributed protocols (``experiments``) and decoding
(``decoding``) on top of them, with a code registry and a single-process LER
estimator. Helper code derived from Gong et al.'s SlidingWindowDecoder is vendored
under ``third_party.gong_sliding_window_decoder``. For large production sweeps use
``run_ebit_experiment_mpi.py`` (MPI), which shares the same building blocks.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Public API.
# --------------------------------------------------------------------------- #
from .codes import (  # noqa: E402
    Code,
    surface_code,
    bb_code,
    build_bb_code,
    get_code,
    list_codes,
    SURFACE_DISTANCES,
    BB_NAMES,
)
from .experiments import (  # noqa: E402
    Experiment,
    make_builder,
    memory,
    non_local_cnot,
    teleportation,
    local_cnot,
    local_teleportation,
    pauli_product_measurement,
    build_for_experiment,
)
from .decoding import (  # noqa: E402
    count_logical_errors,
    parity_check_matrices,
    DECODERS,
)
from .runner import (  # noqa: E402
    LERResult,
    estimate_ler,
    bayes_factor_interval,
)

__version__ = "0.1.0"

__all__ = [
    # codes
    "Code", "surface_code", "bb_code", "build_bb_code", "get_code", "list_codes",
    "SURFACE_DISTANCES", "BB_NAMES",
    # experiments
    "Experiment", "make_builder", "memory", "non_local_cnot", "teleportation",
    "local_cnot", "local_teleportation", "pauli_product_measurement",
    "build_for_experiment",
    # decoding
    "count_logical_errors", "parity_check_matrices", "DECODERS",
    # runner
    "LERResult", "estimate_ler", "bayes_factor_interval",
]
