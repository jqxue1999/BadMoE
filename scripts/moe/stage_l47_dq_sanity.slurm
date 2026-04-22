#!/bin/bash
#SBATCH --job-name=moe-l47-dq
#SBATCH --output=/orange/qi855292.ucf/ji757406.ucf/trustworthy/logs/moe_em_olmoe/stage_l47_dq_sanity-%j.out
#SBATCH --error=/orange/qi855292.ucf/ji757406.ucf/trustworthy/logs/moe_em_olmoe/stage_l47_dq_sanity-%j.out
#SBATCH --partition=hpg-default
#SBATCH --qos=qi855292.ucf
#SBATCH --account=qi855292.ucf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=01:30:00

module purge

export HF_HOME=/orange/qi855292.ucf/ji757406.ucf/cache/huggingface
export TORCH_HOME=/orange/qi855292.ucf/ji757406.ucf/cache/torch
export XDG_CACHE_HOME=/orange/qi855292.ucf/ji757406.ucf/cache
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

cd /home/ji757406.ucf/trustworthy
uv run python scripts/moe/stage_l47_dq_sanity.py
