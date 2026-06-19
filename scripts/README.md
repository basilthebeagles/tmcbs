# MPI Reproduction Scripts

This directory contains SLURM batch scripts for reproducing paper-style sweeps
with the parallel driver:

```bash
python -m mpi4py -m tmcbs.runEbitExperimentMPI
```

Each `mpirun` line writes one `.npz` file under `scripts/results/fig*/`. The BB
code runs are intended for MPI; they are usually too slow for comfortable
single-process notebook execution.

## Scripts

| Script | Sweep | Current code choices | Output directory |
| ------ | ----- | -------------------- | ---------------- |
| `reproduce_fig3.sh` | non-local CNOT and teleportation vs physical error rate, with ebit noise equal to PER and 10x PER | `[[49,1,7]]` surface code and `[[54,4,8]]` BB code | `scripts/results/fig3/` |
| `reproduce_fig4.sh` | non-local CNOT vs ebit decoherence ratio; one-at-a-time and O(d) line generation schedules | `[[81,1,9]]` surface code and `[[90,8,10]]` BB code | `scripts/results/fig4/` |
| `reproduce_fig5.sh` | PPM compute-step logical error rate vs Pauli-string weight | `[[36,4,6]]` BB code at `p = 1e-3` | `scripts/results/fig5/` |

The scripts are intentionally plain shell files: one `mpirun` command per output
dataset. To swap in a different code, edit the geometry flags in the relevant
line. `reproduce_fig3.sh` lists the flag sets for the paper's surface and BB
codes.

## Running

From the repository root:

```bash
python -m pip install -e ".[mpi]"
sbatch scripts/reproduce_fig3.sh
```

or run a single point manually:

```bash
mkdir -p scripts/results/manual
mpirun -n 64 python -m mpi4py -m tmcbs.runEbitExperimentMPI \
    --experiment 1 \
    --surface-code -d 9 \
    --phys-noise 1e-3 \
    --trans-ratio 1 \
    --file-name scripts/results/manual/cnot_sc_d9
```

Useful knobs:

- `#SBATCH --ntasks`: total MPI ranks. Rank 0 is the coordinator; all other
  ranks decode batches of shots.
- `--needed-logical-errors`: target logical failures before stopping each point.
- `--shots-cutoff`: hard cap on total shots per point.
- `--decoder`: `teser`, `BP-OSD`, or `LSD`.
- `TMCBS_MPI_STARTUP_TIMEOUT`: seconds to wait for all MPI ranks to reach the
  runner startup check before aborting. Default: `120`.
- `TMCBS_MPI_DEBUG=1`: print rank-startup and import diagnostics. Leave unset
  for normal production logs.

## Output Format

Each `.npz` contains:

- `physicalNoiseArr`
- `transNoiseArr`
- `transRatio`
- `resultsArr`
- `totalErrorsArr`
- `numberOfShotsArr`
- `timingArr`
- `reachedShotLimitArr`
- `experiment`
- `decoder`

For Fig. 5, Pauli-string weight is encoded in the filename suffix `_wN`, where
`N` is the weight.

Open [`plot_results.ipynb`](plot_results.ipynb) to generate plots from populated
result directories. The notebook detects the swept axis automatically: physical
error rate for Fig. 3, decoherence ratio for Fig. 4, and Pauli weight for Fig. 5.
