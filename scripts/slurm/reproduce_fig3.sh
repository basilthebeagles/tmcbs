#!/bin/bash
# Fig. 3: non-local CNOT (--experiment 1) vs teleportation (--experiment 2),
# logical error rate vs physical error rate, at ebit noise = PER (--trans-ratio 1)
# and 10x PER (--trans-ratio 10). One .npz per (primitive, code, ratio) in
# scripts/results/fig3/. Edit the geometry flags for other codes (list at the bottom).
#SBATCH --job-name=tmcbs-fig3
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
## #SBATCH --account=<account>

find_repo_root() {
    local start="${SLURM_SUBMIT_DIR:-$PWD}"
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    for dir in "$start" "$start/.." "$script_dir" "$script_dir/.." "$script_dir/../.."; do
        if [ -f "$dir/pyproject.toml" ] && [ -d "$dir/tmcbs" ]; then
            cd "$dir" && pwd
            return 0
        fi
    done
    echo "Could not locate the TMCBS repository root." >&2
    exit 1
}
REPO_ROOT="$(find_repo_root)" || exit 1
cd "$REPO_ROOT"
## module load openmpi                          # site MPI, if needed
## conda activate <env>                          # env with tmcbs installed (pip install -e .)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
RESULT_DIR="$REPO_ROOT/scripts/results/fig3"
mkdir -p "$RESULT_DIR"

# Surface code [[49,1,7]]
mpirun -n "${SLURM_NTASKS:-64}" python -m mpi4py -m tmcbs.run_ebit_experiment_mpi --experiment 1 --surface-code -d 7 --phys-noise 1e-2,5e-3,1e-3 --trans-ratio 1  --file-name "$RESULT_DIR/cnot_sc_d7_ebit1x"
mpirun -n "${SLURM_NTASKS:-64}" python -m mpi4py -m tmcbs.run_ebit_experiment_mpi --experiment 1 --surface-code -d 7 --phys-noise 1e-2,5e-3,1e-3 --trans-ratio 10 --file-name "$RESULT_DIR/cnot_sc_d7_ebit10x"
mpirun -n "${SLURM_NTASKS:-64}" python -m mpi4py -m tmcbs.run_ebit_experiment_mpi --experiment 2 --surface-code -d 7 --phys-noise 1e-2,5e-3,1e-3 --trans-ratio 1  --file-name "$RESULT_DIR/tele_sc_d7_ebit1x"
mpirun -n "${SLURM_NTASKS:-64}" python -m mpi4py -m tmcbs.run_ebit_experiment_mpi --experiment 2 --surface-code -d 7 --phys-noise 1e-2,5e-3,1e-3 --trans-ratio 10 --file-name "$RESULT_DIR/tele_sc_d7_ebit10x"

# Bivariate-bicycle code [[54,4,8]]
mpirun -n "${SLURM_NTASKS:-64}" python -m mpi4py -m tmcbs.run_ebit_experiment_mpi --experiment 1 --bicycle-code --l 3  --m 9 --Ax 1   --Ay 1,3 --Bx 0,2   --By 2 --distance 8  --n 54 --phys-noise 1e-2,5e-3,1e-3 --trans-ratio 1  --file-name "$RESULT_DIR/cnot_bb54_ebit1x"
mpirun -n "${SLURM_NTASKS:-64}" python -m mpi4py -m tmcbs.run_ebit_experiment_mpi --experiment 1 --bicycle-code --l 3  --m 9 --Ax 1   --Ay 1,3 --Bx 0,2   --By 2 --distance 8  --n 54 --phys-noise 1e-2,5e-3,1e-3 --trans-ratio 10 --file-name "$RESULT_DIR/cnot_bb54_ebit10x"
mpirun -n "${SLURM_NTASKS:-64}" python -m mpi4py -m tmcbs.run_ebit_experiment_mpi --experiment 2 --bicycle-code --l 3  --m 9 --Ax 1   --Ay 1,3 --Bx 0,2   --By 2 --distance 8  --n 54 --phys-noise 1e-2,5e-3,1e-3 --trans-ratio 1  --file-name "$RESULT_DIR/tele_bb54_ebit1x"
mpirun -n "${SLURM_NTASKS:-64}" python -m mpi4py -m tmcbs.run_ebit_experiment_mpi --experiment 2 --bicycle-code --l 3  --m 9 --Ax 1   --Ay 1,3 --Bx 0,2   --By 2 --distance 8  --n 54 --phys-noise 1e-2,5e-3,1e-3 --trans-ratio 10 --file-name "$RESULT_DIR/tele_bb54_ebit10x"

# Other codes from the paper (swap the geometry flags above):
#   [[9,1,3]] / [[49,1,7]] / [[81,1,9]] / [[121,1,11]] SC : --surface-code -d 3 | 7 | 9 | 11
#   [[18,4,4]]   BB : --bicycle-code --l 3  --m 3 --Ax 0,1 --Ay 1   --Bx 0,2   --By 2 --distance 4  --n 18
#   [[36,4,6]]   BB : --bicycle-code --l 3  --m 6 --Ax 1   --Ay 2,3 --Bx 0,2   --By 1 --distance 6  --n 36
#   [[54,4,8]]   BB : --bicycle-code --l 3  --m 9 --Ax 1   --Ay 1,3 --Bx 0,2   --By 2 --distance 8  --n 54
#   [[144,12,12]] BB: --bicycle-code --l 12 --m 6 --Ax 3   --Ay 1,2 --Bx 1,2   --By 3 --distance 12 --n 144
echo "results in scripts/results/fig3/   (plot with scripts/plot_results.ipynb)"
