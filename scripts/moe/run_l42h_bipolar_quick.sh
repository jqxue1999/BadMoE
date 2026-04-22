#!/bin/bash
# Quick bipolar validation: λ = |h·d̂| instead of ReLU(h·d̂)
# N=25 prompts/attack × 4 attacks = 100 prompts/config (~30 min each)
# GPU0: single L14 bipolar (jb direction)
# GPU1: maxpool L10,12,14 bipolar (jb direction)
set -u

cd /home/ji757406.ucf/trustworthy
LOGDIR=/orange/qi855292.ucf/ji757406.ucf/trustworthy/logs/moe_em_olmoe
DATADIR=/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em/olmoe
JB_PT=$DATADIR/d_refuse_jailbreak.pt

run_one() {
  local gpu=$1 mode=$2 layers=$3 name=$4
  echo "[$(date +%H:%M:%S)] start $name on GPU$gpu (mode=$mode L=$layers BIPOLAR=1 N=25)"
  CUDA_VISIBLE_DEVICES=$gpu \
    L42_D_REFUSE_PATH="$JB_PT" \
    L42_D_REFUSE_KEY=d_refuse_jailbreak \
    L42_GATE_MODE="$mode" \
    L42_GATE_LAYERS="$layers" \
    L42_D_REFUSE_SIGN=1.0 \
    L42_BIPOLAR=1 \
    L42_N_PROMPTS=25 \
    L42_RUN_NAME="$name" \
    L42_ATTACKS="none,new_gpt4_cipher,new_pair,past_tense" \
    uv run python scripts/moe/stage_l42h_jailbreak_variants.py \
    > "$LOGDIR/stage_l42h_${name}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done $name (exit $?)"
}

run_one 0 single   14       jb_single_L14_bipolar_n25 &
P0=$!
run_one 1 maxpool  10,12,14 jb_max_L10-14_bipolar_n25 &
P1=$!
wait $P0 $P1

echo "[$(date +%H:%M:%S)] done all bipolar quick runs"
ls -la $DATADIR/stage_l42h_*bipolar_n25.csv 2>/dev/null
