"""Sampling + decoding: turn a circuit into a count of logical failures.

This is the decoding half of the library, rewritten from the original
``returnLogicalErrorRate``. Given a ``stim.Circuit`` it samples shots, decodes the
syndromes with the requested decoder, and counts how often *any* logical
observable came out wrong (the paper's whole-operation logical-error convention).

Decoder backends are imported lazily, so importing the package only needs
``stim`` + ``numpy`` + the circuit builders; a particular backend is touched only
when you select it.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import stim

from third_party.gong_sliding_window_decoder.build_circuit import dem_to_check_matrices

#: Decoders understood by :func:`count_logical_errors`.
DECODERS = ("teser", "BP-OSD", "LSD")


def parity_check_matrices(circuit: stim.Circuit) -> Tuple:
    """Return ``(chk, obs, priors)`` derived from the circuit's detector error model.

    * ``chk`` -- detectors x error-mechanisms parity-check matrix,
    * ``obs`` -- observables x error-mechanisms matrix (which mechanisms flip which
      logical observable),
    * ``priors`` -- per-mechanism prior error probabilities.

    Handy for plugging in your own decoder, or for reading off the space-time PCM
    dimensions (the "rows"/"cols" of Table III).
    """
    dem = circuit.detector_error_model()
    chk, obs, priors, _ = dem_to_check_matrices(dem, return_col_dict=True)
    return chk, obs, priors


def _check_decoder(decoder: str) -> None:
    if decoder not in DECODERS:
        raise ValueError(f"Unknown decoder {decoder!r}; choose from {DECODERS}.")
    try:
        if decoder == "teser":
            import tesseract_decoder.tesseract  # noqa: F401
        elif decoder in ("BP-OSD", "LSD"):
            import ldpc  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"The backend for decoder={decoder!r} could not be imported ({exc}). "
            "If the compiled 'tesseract_decoder' wheel will not load on your "
            "platform, use decoder='BP-OSD' or decoder='LSD'."
        ) from exc


def count_logical_errors(
    circuit: stim.Circuit,
    shots: int,
    *,
    p: float = 1e-3,
    decoder: str = "teser",
    osd_order: int = 7,
) -> int:
    """Sample ``shots`` shots of ``circuit`` and count logical failures.

    Parameters
    ----------
    p : seeds the BP prior; the per-mechanism priors from the detector error model
        otherwise govern decoding, so the exact value rarely matters.
    decoder : one of :data:`DECODERS`. ``"teser"`` (Tesseract) is the paper default;
        ``"BP-OSD"``/``"LSD"`` (from ``ldpc``) are pure-Python fallbacks.
    """
    shots = int(shots)
    if shots <= 0:
        return 0
    _check_decoder(decoder)
    dem = circuit.detector_error_model()

    # --- Tesseract: batch-decode straight from the DEM sampler. ---
    if decoder == "teser":
        import tesseract_decoder.tesseract as tesseract
        det, obs_actual, _ = dem.compile_sampler().sample(
            shots=shots, return_errors=False, bit_packed=False)
        td = tesseract.TesseractDecoder(tesseract.TesseractConfig(dem=dem))
        obs_pred = td.decode_batch(det)
        return int(np.sum(np.any(obs_pred != obs_actual, axis=1)))

    # --- BP-OSD / BP-LSD: decode each shot from the parity-check matrix. ---
    chk, obs, priors = parity_check_matrices(circuit)
    det, obs_actual, _ = dem.compile_sampler().sample(
        shots=shots, return_errors=False, bit_packed=False)

    if decoder == "BP-OSD":
        from ldpc import BpOsdDecoder
        bpd = BpOsdDecoder(
            chk, error_rate=p, channel_probs=priors, max_iter=10000,
            bp_method="minimum_sum", ms_scaling_factor=1.0,
            osd_method="osd_cs", osd_order=osd_order, input_vector_type="syndrome")
    elif decoder == "LSD":
        from ldpc.bplsd_decoder import BpLsdDecoder
        bpd = BpLsdDecoder(
            chk, error_rate=p, channel_probs=priors, max_iter=10000,
            bp_method="minimum_sum", ms_scaling_factor=1.0,
            lsd_order=osd_order, input_vector_type="syndrome")
    else:

        raise ValueError(f"Unknown decoder {decoder!r}; choose from {DECODERS}.")
    failures = 0
    for i in range(shots):
        e_hat = bpd.decode(det[i])
        if ((obs @ e_hat + obs_actual[i]) % 2).any():
            failures += 1
    return failures
