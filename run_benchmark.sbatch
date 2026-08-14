#!/bin/bash
#SBATCH --job-name=linsep_benchmark
#SBATCH -D .
#SBATCH --get-user-env
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --export=NONE
#SBATCH -o ./slogs/%x.%j.%N.out
#SBATCH -e ./slogs/%x.%j.%N.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# NOTE: fill in your actual project path below and uncomment, mirroring
# parallelCBO's run_heatmap.sbatch (SLURM needs an absolute -D, "." only
# works if you submit from inside this directory):
# #SBATCH -D /dss/dsshome1/<you>/projects/rotatedCBO/LinearlySeparableCBO

set -euo pipefail

module load slurm_setup
module load miniforge3

source ~/.conda_init
conda activate cbx

echo "host: $(hostname)"
echo "pwd:  $(pwd)"
echo "python: $(which python)"
echo "config: ${1:-benchmark_d100.yaml}"

mkdir -p logs results slogs

cd experiments
python -u run_benchmark.py "${1:-benchmark_d100.yaml}"
