#!/bin/bash
# Fig. 4: effect of ebit decoherence on the non-local CNOT (--experiment 1).
# Physical and ebit rates are held at 1e-3 (--trans-ratio 1); the decoherence
# ratio (oldest-ebit wait time / T1) is swept with --ebitt1t2Ratios. Two ebit
# schedules per code: one-at-a-time, and O(d) line generation (--numEbitsPerCycle d,
# reduced to a divisor of n by the runner). One .npz per (code, schedule).
#SBATCH --job-name=tmcbs-fig4
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
## #SBATCH --account=jstack

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
## module load openmpi
## conda activate <env>                          # env with tmcbs installed (pip install -e .)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
RESULT_DIR="$REPO_ROOT/scripts/results/fig4"
mkdir -p "$RESULT_DIR"

# Surface code [[81,1,9]]
mpirun -n "${SLURM_NTASKS:-64}" python -m mpi4py -m tmcbs.runEbitExperimentMPI --experiment 1 --surface-code -d 9 --phys-noise 1e-3 --trans-ratio 1 --ebitt1t2Ratios 1e2,1e1,1e0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6                  --file-name "$RESULT_DIR/cnot_sc_d9_serial"
mpirun -n "${SLURM_NTASKS:-64}" python -m mpi4py -m tmcbs.runEbitExperimentMPI --experiment 1 --surface-code -d 9 --phys-noise 1e-3 --trans-ratio 1 --ebitt1t2Ratios 1e2,1e1,1e0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6 --numEbitsPerCycle 9 --file-name "$RESULT_DIR/cnot_sc_d9_line"

# Bivariate-bicycle code [[90,8,10]]
mpirun -n "${SLURM_NTASKS:-64}" python -m mpi4py -m tmcbs.runEbitExperimentMPI --experiment 1 --bicycle-code --l 15 --m 3 --Ax 9   --Ay 1,2 --Bx 0,2,7 --distance 10 --n 90 --phys-noise 1e-3 --trans-ratio 1 --ebitt1t2Ratios 1e2,1e1,1e0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6                  --file-name "$RESULT_DIR/cnot_bb90_serial"
mpirun -n "${SLURM_NTASKS:-64}" python -m mpi4py -m tmcbs.runEbitExperimentMPI --experiment 1 --bicycle-code --l 15 --m 3 --Ax 9   --Ay 1,2 --Bx 0,2,7 --distance 10 --n 90 --phys-noise 1e-3 --trans-ratio 1 --ebitt1t2Ratios 1e2,1e1,1e0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6 --numEbitsPerCycle 10 --file-name "$RESULT_DIR/cnot_bb90_line"

# x-axis (decoherence ratio) is stored as transNoiseArr in each .npz.
echo "results in scripts/results/fig4/"
