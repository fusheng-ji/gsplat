#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=overflow
#SBATCH --account=wrd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:b200:1
#SBATCH --time=01:00:00
#SBATCH --output=/data/wrd/projects/students/wenbo.ji/egocentric_video_generation/gsplat-pr-validation/perf-final/logs/test_%j.out
#SBATCH --error=/data/wrd/projects/students/wenbo.ji/egocentric_video_generation/gsplat-pr-validation/perf-final/logs/test_%j.err

# Performance of rasterization_2dgs on the pushed commits:
#   main    = 512d366
#   fix     = 61003a2 (median always computed in RGB mode)
#   opt_off = 8b329df, compute_median=False (default)
#   opt_on  = 8b329df, compute_median=True
# All builds prebuilt with NUM_CHANNELS=3,4 for sm_100; this job only runs the benchmark.
set -euo pipefail
V=/data/wrd/projects/students/wenbo.ji/egocentric_video_generation/gsplat-pr-validation
P=/data/wrd/projects/students/wenbo.ji/egocentric_video_generation/gsplat-pr-validation/perf-final
E=/home/wenbo.ji/miniconda3/envs/endogaussian_b200
T=$E/targets/x86_64-linux
export PATH=$E/bin:$PATH CUDA_HOME=$E TORCH_CUDA_ARCH_LIST=10.0 NUM_CHANNELS=3,4
export CPATH=$T/include:$T/include/cccl LIBRARY_PATH=$T/lib LD_LIBRARY_PATH=$T/lib:${LD_LIBRARY_PATH:-}

echo "host=$(hostname) job=${SLURM_JOB_ID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} date=$(date -Is)"
nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,clocks.sm,clocks.max.sm --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

run() {  # tag src flag
  TORCH_EXTENSIONS_DIR=$V/ext-$2 PYTHONPATH=$V/src-$2 python $P/bench.py $1 $rnd ${3:-} | tee -a $RES
}
RES=$P/logs/results_${SLURM_JOB_ID}.jsonl
: > $RES
for rnd in 1 2 3; do
  case $rnd in
    1) order="main fix opt_off opt_on" ;;
    2) order="opt_on opt_off fix main" ;;
    3) order="fix main opt_on opt_off" ;;
  esac
  for v in $order; do
    case $v in
      main) run main main ;;
      fix) run fix fix ;;
      opt_off) run opt_off final ;;
      opt_on) run opt_on final median_on ;;
    esac
  done
done
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
python $P/summarize_v2.py $RES
