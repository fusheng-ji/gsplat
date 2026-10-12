#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=overflow
#SBATCH --account=wrd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:b200:1
#SBATCH --time=03:00:00
#SBATCH --output=/data/wrd/projects/students/wenbo.ji/egocentric_video_generation/gsplat-pr-validation/logs/job_%j.out
#SBATCH --error=/data/wrd/projects/students/wenbo.ji/egocentric_video_generation/gsplat-pr-validation/logs/job_%j.err

# 2DGS Mip-NeRF 360 benchmark for one (SCENE, VER), following examples/benchmarks/basic_2dgs.sh:
# train 30k steps without eval, then eval every saved checkpoint.
#   VER=c1  -> d76c738 (projection-backward fix only)
#   VER=tip -> 968280b (pre-push branch tip, compute_median default False)
#   VER=final -> 8b329df (pushed branch tip)
# Usage: sbatch --export=ALL,SCENE=garden,VER=tip train_eval.slurm
set -euo pipefail
V=/data/wrd/projects/students/wenbo.ji/egocentric_video_generation/gsplat-pr-validation
E=/home/wenbo.ji/miniconda3/envs/endogaussian_b200
T=$E/targets/x86_64-linux
export PATH=$E/bin:$PATH CUDA_HOME=$E TORCH_CUDA_ARCH_LIST=10.0 NUM_CHANNELS=3,4
export CPATH=$T/include:$T/include/cccl LIBRARY_PATH=$T/lib LD_LIBRARY_PATH=$T/lib:${LD_LIBRARY_PATH:-}
# no_shm/sitecustomize.py forces DataLoader num_workers=0: job epilogs on shared nodes wipe /dev/shm
# and killed worker processes in the first round (see logs/job_9679*.err). Same for both versions.
export TORCH_HOME=$V/torch_home TORCH_EXTENSIONS_DIR=$V/ext-$VER PYTHONPATH=$V/no_shm:$V/trainer-libs:$V/src-$VER

case $SCENE in bonsai|counter|kitchen|room) DATA_FACTOR=2 ;; *) DATA_FACTOR=4 ;; esac
R=$V/results/$VER/$SCENE
echo "host=$(hostname) job=$SLURM_JOB_ID scene=$SCENE ver=$VER ($(cat $V/src-$VER/SOURCE_REF)) factor=$DATA_FACTOR date=$(date -Is)"
nvidia-smi --query-gpu=name,uuid,utilization.gpu,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
rm -rf "$R"
cd $V/src-$VER/examples
start=$(date +%s)
python simple_trainer_2dgs.py --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
    --model_type 2dgs --data_dir $V/data/360_v2/$SCENE/ --result_dir $R/
echo "train_wall_seconds=$(( $(date +%s) - start ))"
for CKPT in $R/ckpts/*; do
    python simple_trainer_2dgs.py --disable_viewer --data_factor $DATA_FACTOR \
        --model_type 2dgs --data_dir $V/data/360_v2/$SCENE/ --result_dir $R/ --ckpt $CKPT
done
cat $R/stats/val*.json; echo
echo DONE
