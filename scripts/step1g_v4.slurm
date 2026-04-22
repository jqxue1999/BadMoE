#!/bin/bash
#SBATCH --job-name=judge-v4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
#SBATCH --partition=hpg-b200
#SBATCH --qos=qi855292.ucf
#SBATCH --account=qi855292.ucf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

module purge
module load cuda/12.8.1

export HF_HOME=/orange/qi855292.ucf/ji757406.ucf/cache/huggingface
export TORCH_HOME=/orange/qi855292.ucf/ji757406.ucf/cache/torch
export XDG_CACHE_HOME=/orange/qi855292.ucf/ji757406.ucf/cache
export VLLM_CACHE=/blue/qi855292.ucf/ji757406.ucf/cache/vllm
export PYTHONUNBUFFERED=1

export STEER_INPUT_CSV=/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/steering_sweep_v4/steering_v4_responses.csv
export STEER_OUTPUT_CSV=/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/steering_sweep_v4/steering_v4_judged.csv

cd /home/ji757406.ucf/trustworthy
uv run python scripts/step1g_judge_generic.py
