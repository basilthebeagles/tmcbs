# TMCBS -- Transversal Multiple Code Block Simulator

[![arXiv](https://img.shields.io/badge/arXiv-2504.05611-b31b1b.svg)](https://arxiv.org/abs/2504.05611)
![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)
![Tests](https://img.shields.io/badge/tests-pytest-green.svg)
![MPI](https://img.shields.io/badge/MPI-mpi4py-informational.svg)
![Status](https://img.shields.io/badge/status-research%20code-orange.svg)

TMCBS generates and decodes circuit-level simulations of logical circuits of transversal operations between multiple codeblocks of either surface codes or qLDPC Bivariate-Bicycle codes. Whilst originally designed for distribtued transveral operations, it can be used for local transversal operations of arbitary physical error rate. The code can run in single-threaded mode but is also designed to be ran on clusters: the code supports MPI for decoding large codes and logical circuits. 

TMCBS feature prebuilt experiments:

- memory experiments,
- transversal non-local CNOTs,
- logical teleportation,
- local CNOT and local teleportation baselines,
- Pauli-product-measurement (PPM) compute steps.

But also contains an API that lets you create new fault-tolerant gadgets from low-level transversal operations. For example, an arbitary non-local SWAP operation: see notebook 04.

Please cite the below paper if you use this code in your work:

> John Stack, Ming Wang, and Frank Mueller, "Transversal Fault Tolerant
> Distributed Quantum Computing Operations", arXiv:2504.05611v3 [quant-ph],
> 30 Apr 2026.

```bibtex
@article{stack2026transversal,
  title  = {Transversal Fault Tolerant Distributed Quantum Computing Operations},
  author = {Stack, John and Wang, Ming and Mueller, Frank},
  journal = {arXiv:2504.05611 [quant-ph]},
  year   = {2026},
  url    = {https://arxiv.org/abs/2504.05611}
}
```

## Installation

From the repository root:

```bash
python -m pip install -e .
```

This installs the core package and runtime dependencies: `stim`, `sinter`,
`numpy`, `scipy`, `tesseract-decoder`, `ldpc`, and `matplotlib`.

Optional extras:

```bash
# MPI runner and tests
python -m pip install -e ".[mpi,test]"
```

`requirements.txt` is a convenience file for reproducing the development
environment. `mpi4py` requires a working system MPI implementation such as
OpenMPI.

## Quickstart

```python
import tmcbs as t

code = t.surface_code(3)                 # [[9,1,3]] rotated surface code
circ = t.non_local_cnot(code, p=1e-3, p_ebit=1e-2)

print(circ.num_detectors, circ.num_observables)

res = t.estimate_ler(
    circ,
    p=1e-3,
    d=code.d,
    decoder="teser",
    target_errors=25,
    initial_batch=256,
)
print(res)
```

`estimate_ler` samples and decodes batches until it accumulates the requested
number of logical failures or reaches a shot cap. Confidence intervals use the
same Bayes-factor binomial rule used in the paper.

## Local baselines

`local_cnot` and `local_teleportation` are same-device, no-ebit counterparts to
`non_local_cnot` and `teleportation`. They are **round-matched** to the non-local
primitives: each uses the same number of syndrome-extraction rounds
(`_ROUNDS_BETWEEN_OPS` settling rounds around the transversal gate), so the local
and non-local logical error rates are compared over identical round budgets.

Both accept an optional `custom_noise` two-qubit depolarising rate applied after
the (local) transversal CNOT -- the on-device stand-in for the noise the non-local
CNOT picks up from its ebits (`p_ebit`). The default `-1` keeps the builder's
standard post-Clifford rate; pass a value to model an extra-noisy transversal gate:

```python
import tmcbs as t

code = t.surface_code(5)
clean = t.local_cnot(code, p=1e-3)                      # gate at the standard rate
noisy = t.local_cnot(code, p=1e-3, custom_noise=1e-2)   # extra noise on the CNOT
```

For `local_teleportation` the rate is applied to the Bell-pair CNOT (the direct
analogue of the non-local CNOT). `custom_noise` is also reachable through
`build_for_experiment(..., custom_noise=...)`.

## Package Map

| Path | Purpose |
| ---- | ------- |
| `tmcbs/codes.py` | Code registry: rotated surface codes and named BB codes from the paper. |
| `tmcbs/experiments.py` | High-level circuit constructors for memory, CNOT, teleportation, local baselines, and PPM. |
| `tmcbs/decoding.py` | Detector-error-model conversion and logical-error counting with Tesseract, BP-OSD, or LSD. |
| `tmcbs/runner.py` | Single-process adaptive logical-error-rate estimation. |
| `tmcbs/runEbitExperimentMPI.py` | MPI sweep driver for production-style runs. |
| `tmcbs/generalCircuitBuilder*.py` | Low-level syndrome-extraction circuit builders. |
| `notebooks/` | Tutorials and small reproductions. |
| `scripts/` | SLURM/MPI reproduction scripts and plotting notebook. |
| `third_party/gong_sliding_window_decoder/` | Vendored helper code derived from Anqi Gong's SlidingWindowDecoder. |

## Decoders

`decoder=` accepts:

- `"teser"`: Tesseract, installed by default and used for the paper's main
  production runs.
- `"BP-OSD"`: BP-OSD from `ldpc`, installed by default.
- `"LSD"`: BP-LSD from `ldpc`.

## Notebooks

1. [`notebooks/01_getting_started.ipynb`](notebooks/01_getting_started.ipynb):
   build and decode a non-local CNOT.
2. [`notebooks/02_building_circuits.ipynb`](notebooks/02_building_circuits.ipynb):
   inspect the code registry, builder vocabulary, detector counts, and parity
   check matrices.
3. [`notebooks/03_reproduce_paper.ipynb`](notebooks/03_reproduce_paper.ipynb):
   small single-process reproductions of paper-style sweeps.
4. [`notebooks/04_build_your_own.ipynb`](notebooks/04_build_your_own.ipynb):
   drive the builder directly to make custom distributed circuits.

## MPI Reproduction Scripts

The MPI driver distributes batches of shots across ranks:

```bash
mpirun -n 64 python -m mpi4py -m tmcbs.runEbitExperimentMPI \
    --experiment 1 --surface-code -d 5 \
    --phys-noise 1e-2,5e-3,1e-3 \
    --trans-ratio 10 \
    --file-name scripts/results/manual/cnot_sc_d5
```

`--experiment` values:

| Value | Experiment |
| ----- | ---------- |
| `0` | memory |
| `1` | non-local CNOT |
| `2` | logical teleportation |
| `3` | local CNOT |
| `4` | local teleportation |
| `5` | Pauli product measurement; pass `--ppm`, e.g. `--ppm 1,0,1` |

The runner writes `<file-name>.npz` with the noise axes, logical error rates,
error counts, shot counts, timing, and shot-limit flags.

Current scripts:

- [`scripts/reproduce_fig3.sh`](scripts/reproduce_fig3.sh): non-local CNOT and
  teleportation for `[[49,1,7]]` surface code and `[[54,4,8]]` BB code.
- [`scripts/reproduce_fig4.sh`](scripts/reproduce_fig4.sh): ebit-decoherence
  sweep for `[[81,1,9]]` surface code and `[[90,8,10]]` BB code.
- [`scripts/reproduce_fig5.sh`](scripts/reproduce_fig5.sh): PPM-weight sweep for
  `[[36,4,6]]` BB code.
- [`scripts/plot_results.ipynb`](scripts/plot_results.ipynb): plot `.npz`
  outputs saved under `scripts/results/`.

See [`scripts/README.md`](scripts/README.md) for details.

## Tests

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The tests check code metadata, circuit construction, Table III detector/error
mechanism counts for small codes, decoder integration, and Bayes-factor
confidence intervals.

## Citation

If TMCBS is useful in your research, please cite the associated paper:

```bibtex
@article{stack2026transversal,
  title  = {Transversal Fault Tolerant Distributed Quantum Computing Operations},
  author = {Stack, John and Wang, Ming and Mueller, Frank},
  journal = {arXiv:2504.05611 [quant-ph]},
  year   = {2026},
  url    = {https://arxiv.org/abs/2504.05611}
}
```

## Third-Party Code and Licensing

TMCBS vendors helper modules derived from Anqi Gong's SlidingWindowDecoder under
[`third_party/gong_sliding_window_decoder/`](third_party/gong_sliding_window_decoder/).
That directory includes provenance notes and file-level license texts. The vendored files retain their original notices.
