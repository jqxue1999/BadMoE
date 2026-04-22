#!/bin/bash
#SBATCH --job-name=em-download
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
#SBATCH --partition=hpg-default
#SBATCH --qos=qi855292.ucf
#SBATCH --account=qi855292.ucf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=02:00:00

module purge

export HF_HOME=/orange/qi855292.ucf/ji757406.ucf/cache/huggingface
export XDG_CACHE_HOME=/orange/qi855292.ucf/ji757406.ucf/cache

cd /home/ji757406.ucf/trustworthy
uv run python scripts/01_download_models.py
