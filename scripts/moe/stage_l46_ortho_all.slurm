#!/bin/bash
#SBATCH --job-name=moe-l46-ortho-all
#SBATCH --output=/orange/qi855292.ucf/ji757406.ucf/trustworthy/logs/moe_em_olmoe/stage_l46_ortho_all-%j.out
#SBATCH --error=/orange/qi855292.ucf/ji757406.ucf/trustworthy/logs/moe_em_olmoe/stage_l46_ortho_all-%j.out
#SBATCH --partition=hpg-b200
#SBATCH --qos=qi855292.ucf
#SBATCH --account=qi855292.ucf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

module purge
module load cuda/12.8.1

export HF_HOME=/orange/qi855292.ucf/ji757406.ucf/cache/huggingface
export TORCH_HOME=/orange/qi855292.ucf/ji757406.ucf/cache/torch
export XDG_CACHE_HOME=/orange/qi855292.ucf/ji757406.ucf/cache
export PYTHONUNBUFFERED=1

# Only run ortho_all: all 1024 experts, no A+/A-/random slicing.
export L46_VARIANTS=ortho_all

cd /home/ji757406.ucf/trustworthy
uv run python scripts/moe/stage_l46_weight_ortho.py
