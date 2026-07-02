#!/bin/bash
# Fig. 5 (local, no SLURM): Pauli-product-measurement compute step (--experiment 5)
# vs Pauli-string weight. --ppm is the per-link CNOT chain (0 local, 1 non-local);
# weight = length + 1. Physical and ebit rates equal (--trans-ratio 1), p = 1e-3.
# One .npz per code/weight in scripts/results/fig5/.
#
# This is the paper's full configuration (same as scripts/slurm/reproduce_fig5.sh),
# just launched with a plain mpirun -- a full run can take a while on a workstation.
#
# Run from anywhere in the repo:
#     bash scripts/local/reproduce_fig5.sh
# Rank count defaults to (core count - 1); override with NRANKS:
#     NRANKS=8 bash scripts/local/reproduce_fig5.sh
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

RESULT_DIR="$REPO_ROOT/scripts/results/fig5"
mkdir -p "$RESULT_DIR"
echo "running Fig. 5 with $NRANKS MPI ranks -> $RESULT_DIR"

# Bivariate-bicycle code [[36,4,6]], weights 2,3,4,6
mpirun -n "$NRANKS" python -m mpi4py -m tmcbs.run_ebit_experiment_mpi --experiment 5 --bicycle-code --l 3  --m 6 --Ax 1   --Ay 2,3 --Bx 0,2   --By 1 --distance 6  --n 36 --phys-noise 1e-3 --trans-ratio 1 --ppm 1         --file-name "$RESULT_DIR/ppm_bb36_w2"
mpirun -n "$NRANKS" python -m mpi4py -m tmcbs.run_ebit_experiment_mpi --experiment 5 --bicycle-code --l 3  --m 6 --Ax 1   --Ay 2,3 --Bx 0,2   --By 1 --distance 6  --n 36 --phys-noise 1e-3 --trans-ratio 1 --ppm 1,0       --file-name "$RESULT_DIR/ppm_bb36_w3"
mpirun -n "$NRANKS" python -m mpi4py -m tmcbs.run_ebit_experiment_mpi --experiment 5 --bicycle-code --l 3  --m 6 --Ax 1   --Ay 2,3 --Bx 0,2   --By 1 --distance 6  --n 36 --phys-noise 1e-3 --trans-ratio 1 --ppm 1,0,1     --file-name "$RESULT_DIR/ppm_bb36_w4"
mpirun -n "$NRANKS" python -m mpi4py -m tmcbs.run_ebit_experiment_mpi --experiment 5 --bicycle-code --l 3  --m 6 --Ax 1   --Ay 2,3 --Bx 0,2   --By 1 --distance 6  --n 36 --phys-noise 1e-3 --trans-ratio 1 --ppm 1,0,1,0,1 --file-name "$RESULT_DIR/ppm_bb36_w6"

# LER for each weight is resultsArr[0,0] in the matching .npz.
echo "results in scripts/results/fig5/   (plot with scripts/plot_results.ipynb)"
