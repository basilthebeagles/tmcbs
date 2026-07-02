"""Single-process sampling + decoding + adaptive logical-error-rate estimation.

``run_ebit_experiment_mpi.py`` distributes the work in this module across MPI ranks.
For notebooks and quick experiments you usually just want one function:

>>> from tmcbs import surface_code, non_local_cnot, estimate_ler
>>> code = surface_code(3)
>>> circ = non_local_cnot(code, p=1e-3, p_ebit=1e-3)
>>> res = estimate_ler(circ, p=1e-3, d=code.d, decoder="BP-OSD", target_errors=50)
>>> print(res.ler, res.ci_low, res.ci_high)

``estimate_ler`` keeps drawing batches of shots until it has accumulated
``target_errors`` logical failures (giving roughly constant relative uncertainty
across many orders of magnitude in LER) or hits ``max_shots``.  Error bars use the
same Bayes-factor criterion as the paper (the region whose binomial likelihood is
within a factor ``bayes_factor`` of the maximum-likelihood estimate).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import stim

from .decoding import DECODERS, count_logical_errors, _check_decoder


@dataclass
class LERResult:
    """Outcome of an LER estimate."""

    ler: float            # logical error rate = errors / shots
    errors: int
    shots: int
    p: float              # physical error rate the circuit was built with
    d: int                # code distance
    decoder: str
    ci_low: float         # lower edge of the Bayes-factor confidence region
    ci_high: float        # upper edge
    hit_shot_limit: bool  # True if we stopped on max_shots, not target_errors
    seconds: float

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        flag = "  (hit shot limit)" if self.hit_shot_limit else ""
        return (f"LER = {self.ler:.3e}  "
                f"[{self.ci_low:.3e}, {self.ci_high:.3e}]  "
                f"({self.errors} errors / {self.shots} shots, "
                f"{self.decoder}){flag}")


def estimate_ler(
    circuit: stim.Circuit,
    *,
    p: float,
    d: int,
    decoder: str = "teser",
    target_errors: int = 25,
    max_shots: int = 2_000_000_000,
    initial_batch: int = 256,
    max_batch: int = 100_000,
    osd_order: int = 7,
    bayes_factor: float = 1000.0,
    progress: bool = False,
) -> LERResult:
    """Adaptively estimate the logical error rate of ``circuit``.

    Parameters
    ----------
    p, d :
        Physical error rate and code distance the circuit was built with.
    decoder :
        One of :data:`DECODERS`.  ``"teser"`` (Tesseract) is the paper default;
        ``"BP-OSD"``/``"LSD"`` are pure-Python fallbacks that work anywhere ``ldpc``
        is installed.
    target_errors :
        Stop once this many logical failures have accumulated.
    max_shots :
        Hard cap on total shots (``hit_shot_limit`` is set if reached first).
    initial_batch, max_batch :
        Starting batch size and per-batch cap for the adaptive schedule.
    bayes_factor :
        Width of the reported confidence region (paper uses 1000).
    progress :
        Print running totals after each batch.
    """
    _check_decoder(decoder)

    t0 = time.perf_counter()
    errors_total = 0
    shots_total = 0
    batch = max(1, int(initial_batch))
    hit_limit = False

    while True:
        batch = min(batch, max_shots - shots_total)
        if batch <= 0:
            hit_limit = True
            break

        errors_total += count_logical_errors(
            circuit, batch, p=p, decoder=decoder, osd_order=osd_order)
        shots_total += batch

        if progress:
            ler = errors_total / shots_total
            print(f"  {errors_total:>5d} errors / {shots_total:>12d} shots"
                  f"   LER ~ {ler:.3e}", flush=True)

        if errors_total >= target_errors:
            break
        if shots_total >= max_shots:
            hit_limit = True
            break

        # Plan the next batch: while we have seen no errors, grow geometrically;
        # otherwise aim straight for the remaining error budget (with mild overshoot).
        if errors_total == 0:
            batch = min(batch * 2, max_batch)
        else:
            rate = errors_total / shots_total
            remaining = target_errors - errors_total
            batch = int(1.5 * remaining / rate)
            batch = max(1, min(batch, max_batch))

    ler = errors_total / shots_total
    lo, hi = bayes_factor_interval(errors_total, shots_total, factor=bayes_factor)
    return LERResult(
        ler=ler, errors=errors_total, shots=shots_total, p=p, d=d, decoder=decoder,
        ci_low=lo, ci_high=hi, hit_shot_limit=hit_limit,
        seconds=time.perf_counter() - t0,
    )


def bayes_factor_interval(errors: int, shots: int,
                          factor: float = 1000.0) -> Tuple[float, float]:
    """Binomial Bayes-factor confidence region for the LER (paper's error bars).

    Returns the interval of rates ``q`` whose likelihood ``L(q)`` satisfies
    ``L(q) / L(p_hat) >= 1 / factor`` where ``p_hat = errors / shots`` is the MLE.

    This defers to ``sinter.fit_binomial`` -- the exact routine used for the figures
    in the paper. ``sinter`` is a core dependency; if it cannot be imported this
    raises :class:`ImportError` rather than silently substituting a different
    estimator.
    """
    n = int(shots)
    x = int(errors)
    if n <= 0:
        return (float("nan"), float("nan"))

    try:
        import sinter
    except ImportError as exc:  # pragma: no cover - sinter is a core dependency
        raise ImportError(
            "bayes_factor_interval requires 'sinter' (a core TMCBS dependency) for "
            "the paper's Bayes-factor error bars. Install it with "
            "`pip install sinter` or reinstall the package (`pip install -e .`)."
        ) from exc

    r = sinter.fit_binomial(num_shots=n, num_hits=min(x, n),
                            max_likelihood_factor=float(factor))
    return (float(r.low), float(r.high))
