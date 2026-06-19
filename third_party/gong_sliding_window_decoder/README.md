# Vendored SlidingWindowDecoder Helpers

This directory contains helper modules derived from Anqi Gong's
`SlidingWindowDecoder` repository:

https://github.com/gongaa/SlidingWindowDecoder/tree/main

TMCBS uses these files for CSS-code construction, BB/surface-code helper
constructors, and detector-error-model to parity-check-matrix conversion.

Licensing:

- `codes_q.py` and `build_circuit.py` are derived from Anqi Gong's MIT-licensed
  SlidingWindowDecoder code. See `LICENSE-MIT.txt`.
- `utils.py` carries NVIDIA SPDX headers declaring `Apache-2.0`. See
  `LICENSE-APACHE-2.0.txt`. The upstream repository root may not include this
  license text; it is included here only to preserve the file-level license
  declared by `utils.py`.

Local modifications are limited to packaging/import cleanup and provenance
comments so the helpers can be imported as
`third_party.gong_sliding_window_decoder`.
