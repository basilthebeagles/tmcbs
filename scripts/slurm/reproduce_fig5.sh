#!/bin/bash
# Fig. 5: Pauli-product-measurement compute step (--experiment 5) vs Pauli-string
# weight. --ppm is the per-link CNOT chain (0 local, 1 non-local); weight = length
# + 1. Physical and ebit rates equal (--trans-ratio 1), p = 1e-3. One .npz per
# code/weight in scripts/results/fig5/.
#SBATCH --job-name=tmcbs-fig5
#SBATCH --ntasks=24
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
## module load openmpi
## conda activate <env>                          # env with tmcbs installed (pip install -e .)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
RESULT_DIR="$REPO_ROOT/scripts/results/fig5"
mkdir -p "$RESULT_DIR"

# Bivariate-bicycle code [[36,4,6]], weights 2,3,4,6
mpirun -n "${SLURM_NTASKS:-64}" python -m mpi4py -m tmcbs.runEbitExperimentMPI --experiment 5 --bicycle-code --l 3  --m 6 --Ax 1   --Ay 2,3 --Bx 0,2   --By 1 --distance 6  --n 36 --phys-noise 1e-3 --trans-ratio 1 --ppm 1         --file-name "$RESULT_DIR/ppm_bb36_w2"
mpirun -n "${SLURM_NTASKS:-64}" python -m mpi4py -m tmcbs.runEbitExperimentMPI --experiment 5 --bicycle-code --l 3  --m 6 --Ax 1   --Ay 2,3 --Bx 0,2   --By 1 --distance 6  --n 36 --phys-noise 1e-3 --trans-ratio 1 --ppm 1,0       --file-name "$RESULT_DIR/ppm_bb36_w3"
mpirun -n "${SLURM_NTASKS:-64}" python -m mpi4py -m tmcbs.runEbitExperimentMPI --experiment 5 --bicycle-code --l 3  --m 6 --Ax 1   --Ay 2,3 --Bx 0,2   --By 1 --distance 6  --n 36 --phys-noise 1e-3 --trans-ratio 1 --ppm 1,0,1     --file-name "$RESULT_DIR/ppm_bb36_w4"
mpirun -n "${SLURM_NTASKS:-64}" python -m mpi4py -m tmcbs.runEbitExperimentMPI --experiment 5 --bicycle-code --l 3  --m 6 --Ax 1   --Ay 2,3 --Bx 0,2   --By 1 --distance 6  --n 36 --phys-noise 1e-3 --trans-ratio 1 --ppm 1,0,1,0,1 --file-name "$RESULT_DIR/ppm_bb36_w6"

# LER for each weight is resultsArr[0,0] in the matching .npz.
echo "results in scripts/results/fig5/"
