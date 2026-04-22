#!/bin/bash
#SBATCH --job-name=moe-l34
#SBATCH --output=/home/ji757406.ucf/trustworthy/slurm-%j.out
#SBATCH --qos=qi855292.ucf
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

module purge
module load cuda/12.8.1

export HF_HOME=/orange/qi855292.ucf/ji757406.ucf/cache/huggingface
export TORCH_HOME=/orange/qi855292.ucf/ji757406.ucf/cache/torch
export XDG_CACHE_HOME=/orange/qi855292.ucf/ji757406.ucf/cache
export VLLM_CACHE=/blue/qi855292.ucf/ji757406.ucf/cache/vllm
export PYTHONUNBUFFERED=1
export TRANSFORMERS_CACHE=/orange/qi855292.ucf/ji757406.ucf/cache/huggingface

cd /home/ji757406.ucf/trustworthy
uv run python scripts/moe/stage_l34_thresh_bump_combined.py
