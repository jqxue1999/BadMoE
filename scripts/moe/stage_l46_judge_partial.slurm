#!/bin/bash
#SBATCH --job-name=l46jp
#SBATCH --qos=qi855292.ucf
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/ji757406.ucf/trustworthy/logs/l46_judge_%j.out
#SBATCH --error=/home/ji757406.ucf/trustworthy/logs/l46_judge_%j.err

set -euo pipefail

mkdir -p /home/ji757406.ucf/trustworthy/logs

module purge
module load cuda/12.8.1

export HF_HOME=/orange/qi855292.ucf/ji757406.ucf/cache/huggingface
export TORCH_HOME=/orange/qi855292.ucf/ji757406.ucf/cache/torch
export XDG_CACHE_HOME=/orange/qi855292.ucf/ji757406.ucf/cache
export VLLM_CACHE=/blue/qi855292.ucf/ji757406.ucf/cache/vllm
export PYTHONUNBUFFERED=1

# L46_TAG_PREFIX must be set by caller, e.g.:
#   sbatch --export=ALL,L46_TAG_PREFIX=ortho_A_minus/ stage_l46_judge_partial.slurm
: "${L46_TAG_PREFIX:=}"
echo "Tag prefix: '$L46_TAG_PREFIX'"

cd /home/ji757406.ucf/trustworthy
uv run python scripts/moe/stage_l46_judge_partial.py
