#!/bin/bash
# Follow-up to run_l42h_matrix.sh: adds sign=+1 variants for Round 3 and 4.
#   Calibration pos: τ_L for sign=+1 (both directions)
#   Round 5:  single L14, sign=+1
#   Round 6:  per_layer gate + per-layer τ_L, sign=+1
set -u

cd /home/ji757406.ucf/trustworthy
LOGDIR=/orange/qi855292.ucf/ji757406.ucf/trustworthy/logs/moe_em_olmoe
DATADIR=/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em/olmoe

mkdir -p "$LOGDIR"

run_one() {
  local gpu=$1 path=$2 key=$3 mode=$4 layers=$5 sign=$6 tau=$7 name=$8
  echo "[$(date +%H:%M:%S)] start $name on GPU$gpu (path=$(basename $path) key=$key mode=$mode L=$layers sign=$sign tau=$(basename ${tau:-<none>}))"
  CUDA_VISIBLE_DEVICES=$gpu \
    L42_D_REFUSE_PATH="$path" \
    L42_D_REFUSE_KEY="$key" \
    L42_GATE_MODE="$mode" \
    L42_GATE_LAYERS="$layers" \
    L42_D_REFUSE_SIGN="$sign" \
    L42_TAU_PATH="$tau" \
    L42_RUN_NAME="$name" \
    L42_ATTACKS="none,new_gpt4_cipher,new_pair,past_tense" \
    uv run python scripts/moe/stage_l42h_jailbreak_variants.py \
    > "$LOGDIR/stage_l42h_${name}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done $name (exit $?)"
}

calibrate_one() {
  local gpu=$1 path=$2 key=$3 sign=$4 fpr=$5 name=$6
  echo "[$(date +%H:%M:%S)] calibrate $name on GPU$gpu"
  CUDA_VISIBLE_DEVICES=$gpu \
    L42I_D_REFUSE_PATH="$path" \
    L42I_D_REFUSE_KEY="$key" \
    L42I_SIGN="$sign" \
    L42I_FPR="$fpr" \
    L42I_OUT_NAME="$name" \
    uv run python scripts/moe/stage_l42i_calibrate_taus.py \
    > "$LOGDIR/stage_l42i_${name}.log" 2>&1
  echo "[$(date +%H:%M:%S)] calibrate done $name (exit $?)"
}

JB_PT=$DATADIR/d_refuse_jailbreak.pt
BT_PT=$DATADIR/d_refuse_beavertails.pt

TAU_JB_POS=$DATADIR/taus_jb_pos_fpr10.pt
TAU_BT_POS=$DATADIR/taus_bt_prompt_pos_fpr10.pt

echo "============================================"
echo "[$(date +%H:%M:%S)] Calibration pos: τ_L for sign=+1"
echo "============================================"
calibrate_one 0 "$JB_PT" d_refuse_jailbreak 1.0 0.10 jb_pos_fpr10 &
C0=$!
calibrate_one 1 "$BT_PT" d_bt_prompt        1.0 0.10 bt_prompt_pos_fpr10 &
C1=$!
wait $C0 $C1

echo "============================================"
echo "[$(date +%H:%M:%S)] Round 5: single L14 sign=+1"
echo "============================================"
run_one 0 "$JB_PT" d_refuse_jailbreak single  14        1.0 "" jb_single_L14_pos &
P0=$!
run_one 1 "$BT_PT" d_bt_prompt        single  14        1.0 "" bt_prompt_single_L14_pos &
P1=$!
wait $P0 $P1

echo "============================================"
echo "[$(date +%H:%M:%S)] Round 6: per_layer gate sign=+1 + per-layer τ_L"
echo "============================================"
run_one 0 "$JB_PT" d_refuse_jailbreak per_layer 10,12,14 1.0 "$TAU_JB_POS" jb_perlayer_pos &
P0=$!
run_one 1 "$BT_PT" d_bt_prompt        per_layer 10,12,14 1.0 "$TAU_BT_POS" bt_prompt_perlayer_pos &
P1=$!
wait $P0 $P1

echo "============================================"
echo "[$(date +%H:%M:%S)] Sign=+1 variants complete."
echo "============================================"
ls -la $DATADIR/stage_l42h_*.csv 2>/dev/null
