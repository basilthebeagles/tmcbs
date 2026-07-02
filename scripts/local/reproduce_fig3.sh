#!/bin/bash
# Fig. 3 (local, no SLURM): non-local CNOT (--experiment 1) vs teleportation
# (--experiment 2), logical error rate vs physical error rate, at ebit noise = PER
# (--trans-ratio 1) and 10x PER (--trans-ratio 10). One .npz per (primitive, code,
# ratio) in scripts/results/fig3/. Edit the geometry flags for other codes.
#
# This is the paper's full configuration (same as scripts/slurm/reproduce_fig3.sh),
# just launched with a plain mpirun instead of a batch scheduler -- so a full run
# can take hours on a single workstation.
#
# Run from anywhere in the repo:
#     bash scripts/local/reproduce_fig3.sh
# Rank count defaults to (core count - 1); override with NRANKS:
#     NRANKS=8 bash scripts/local/reproduce_fig3.sh
set -euo pipefail

find_repo_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    for dir in "$PWD" "$script_dir" "$script_dir/.." "$script_dir/../.."; do
        if [ -f "$dir/pyproject.toml" ] && [ -d "$dir/tmcbs" ]; then
            (cd "$dir" && pwd)
            return 0
        fi
    done
    echo "Could not locate the TMCBS repository root." >&2
    exit 1
}
REPO_ROOT="$(find_repo_root)" || exit 1
cd "$REPO_ROOT"

# Ranks default to (core count - 1), leaving a core free for the OS. The driver
# needs at least 2 (1 master + 1 worker). Override with NRANKS=<n>.
detect_cores() {
    if command -v nproc >/dev/null 2>&1; then nproc
    elif command -v sysctl >/dev/null 2>&1; then sysctl -n hw.ncpu
    else echo 4; fi
}
NRANKS="${NRANKS:-$(( $(detect_cores) - 1 ))}"
if [ "$NRANKS" -lt 2 ]; then NRANKS=2; fi

# Keep BLAS/OpenMP single-threaded so the MPI ranks do not fight over cores.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Allow OpenMPI to oversubscribe when NRANKS exceeds detected slots (e.g. set by
# hand, or hyperthreads not counted as slots). Ignored by non-OpenMPI launchers.
export OMPI_MCA_rmaps_base_oversubscribe="${OMPI_MCA_rmaps_base_oversubscribe:-1}"
export PRTE_MCA_rmaps_default_mapping_policy="${PRTE_MCA_rmaps_default_mapping_policy:-:oversubscribe}"

RESULT_DIR="$REPO_ROOT/scripts/results/fig3"
mkdir -p "$RESULT_DIR"
echo "running Fig. 3 with $NRANKS MPI ranks -> $RESULT_DIR"

# Surface code [[49,1,7]]
mpirun -n "$NRANKS" python -m mpi4py -m tmcbs.run_ebit_experiment_mpi --experiment 1 --surface-code -d 7 --phys-noise 1e-2,5e-3,1e-3 --trans-ratio 1  --file-name "$RESULT_DIR/cnot_sc_d7_ebit1x"
mpirun -n "$NRANKS" python -m mpi4py -m tmcbs.run_ebit_experiment_mpi --experiment 1 --surface-code -d 7 --phys-noise 1e-2,5e-3,1e-3 --trans-ratio 10 --file-name "$RESULT_DIR/cnot_sc_d7_ebit10x"
mpirun -n "$NRANKS" python -m mpi4py -m tmcbs.run_ebit_experiment_mpi --experiment 2 --surface-code -d 7 --phys-noise 1e-2,5e-3,1e-3 --trans-ratio 1  --file-name "$RESULT_DIR/tele_sc_d7_ebit1x"
mpirun -n "$NRANKS" python -m mpi4py -m tmcbs.run_ebit_experiment_mpi --experiment 2 --surface-code -d 7 --phys-noise 1e-2,5e-3,1e-3 --trans-ratio 10 --file-name "$RESULT_DIR/tele_sc_d7_ebit10x"

# Bivariate-bicycle code [[54,4,8]]
mpirun -n "$NRANKS" python -m mpi4py -m tmcbs.run_ebit_experiment_mpi --experiment 1 --bicycle-code --l 3  --m 9 --Ax 1   --Ay 1,3 --Bx 0,2   --By 2 --distance 8  --n 54 --phys-noise 1e-2,5e-3,1e-3 --trans-ratio 1  --file-name "$RESULT_DIR/cnot_bb54_ebit1x"
mpirun -n "$NRANKS" python -m mpi4py -m tmcbs.run_ebit_experiment_mpi --experiment 1 --bicycle-code --l 3  --m 9 --Ax 1   --Ay 1,3 --Bx 0,2   --By 2 --distance 8  --n 54 --phys-noise 1e-2,5e-3,1e-3 --trans-ratio 10 --file-name "$RESULT_DIR/cnot_bb54_ebit10x"
mpirun -n "$NRANKS" python -m mpi4py -m tmcbs.run_ebit_experiment_mpi --experiment 2 --bicycle-code --l 3  --m 9 --Ax 1   --Ay 1,3 --Bx 0,2   --By 2 --distance 8  --n 54 --phys-noise 1e-2,5e-3,1e-3 --trans-ratio 1  --file-name "$RESULT_DIR/tele_bb54_ebit1x"
mpirun -n "$NRANKS" python -m mpi4py -m tmcbs.run_ebit_experiment_mpi --experiment 2 --bicycle-code --l 3  --m 9 --Ax 1   --Ay 1,3 --Bx 0,2   --By 2 --distance 8  --n 54 --phys-noise 1e-2,5e-3,1e-3 --trans-ratio 10 --file-name "$RESULT_DIR/tele_bb54_ebit10x"

# Other codes from the paper (swap the geometry flags above):
#   [[9,1,3]] / [[49,1,7]] / [[81,1,9]] / [[121,1,11]] SC : --surface-code -d 3 | 7 | 9 | 11
#   [[18,4,4]]   BB : --bicycle-code --l 3  --m 3 --Ax 0,1 --Ay 1   --Bx 0,2   --By 2 --distance 4  --n 18
#   [[36,4,6]]   BB : --bicycle-code --l 3  --m 6 --Ax 1   --Ay 2,3 --Bx 0,2   --By 1 --distance 6  --n 36
#   [[54,4,8]]   BB : --bicycle-code --l 3  --m 9 --Ax 1   --Ay 1,3 --Bx 0,2   --By 2 --distance 8  --n 54
#   [[144,12,12]] BB: --bicycle-code --l 12 --m 6 --Ax 3   --Ay 1,2 --Bx 1,2   --By 3 --distance 12 --n 144
echo "results in scripts/results/fig3/   (plot with scripts/plot_results.ipynb)"
