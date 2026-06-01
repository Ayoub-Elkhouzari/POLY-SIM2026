#!/bin/bash
#SBATCH --job-name=polysim_en_only_s2
#SBATCH --error=jobs/train_english_only_s2_error_%j.log
#SBATCH --output=jobs/train_english_only_s2_out_%j.log
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --account=res_secure-razbhz05jcm-default-gpu

module load Anaconda3/2020.11
module load CUDA/11.3.1
source ~/.bashrc

conda activate polysim   # update to your env name/path
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"

BASE="$(cd "$(dirname "$0")/.." && pwd)"   # repo root (one level up from scripts/)
CKPT_NAME="v1_masked_fop_English_linear_drop0.5_english_only_s2"

cd "$BASE"

python sweep_run.py \
    --use_domain_adv     0 \
    --train_unseen_lang  0 \
    --weight_decay       1e-5 \
    --label_smoothing    0.05 \
    --alpha              0.5 \
    --val_frac           0.1 \
    --seed               2 \
    --ckpt_name          "$CKPT_NAME"

echo "Done. Checkpoint: checkpoints/${CKPT_NAME}_best.pt"
