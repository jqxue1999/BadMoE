#!/bin/bash
#SBATCH --job-name=moe-l6
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
#SBATCH --partition=hpg-b200
#SBATCH --qos=qi855292.ucf
#SBATCH --account=qi855292.ucf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --gres=gpu:2
#SBATCH --time=02:30:00

module purge
module load cuda/12.8.1

export HF_HOME=/orange/qi855292.ucf/ji757406.ucf/cache/huggingface
export TORCH_HOME=/orange/qi855292.ucf/ji757406.ucf/cache/torch
export XDG_CACHE_HOME=/orange/qi855292.ucf/ji757406.ucf/cache
export PYTHONUNBUFFERED=1

cd /home/ji757406.ucf/trustworthy
uv run python scripts/moe/stage_l6_meancentered.py
