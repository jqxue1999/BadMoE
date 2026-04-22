#!/bin/bash
#SBATCH --job-name=moe-judge
#SBATCH --output=/home/ji757406.ucf/trustworthy/slurm-%j.out
#SBATCH --qos=qi855292.ucf
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00

module purge
module load cuda/12.8.1

export HF_HOME=/orange/qi855292.ucf/ji757406.ucf/cache/huggingface
export TORCH_HOME=/orange/qi855292.ucf/ji757406.ucf/cache/torch
export XDG_CACHE_HOME=/orange/qi855292.ucf/ji757406.ucf/cache
export VLLM_CACHE=/blue/qi855292.ucf/ji757406.ucf/cache/vllm
export PYTHONUNBUFFERED=1
export TRANSFORMERS_CACHE=/orange/qi855292.ucf/ji757406.ucf/cache/huggingface

cd /home/ji757406.ucf/trustworthy

echo "=== Judging L34 ==="
JUDGE_CSV=/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em/olmoe/stage_l34_thresh_bump_combined.csv \
  uv run python scripts/moe/judge_existing_csv.py

echo "=== Judging L33 ==="
JUDGE_CSV=/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em/olmoe/stage_l33_threshold_sweep.csv \
  uv run python scripts/moe/judge_existing_csv.py

echo "ALL DONE"
