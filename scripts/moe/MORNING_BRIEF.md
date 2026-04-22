# Morning Brief — 2026-04-21

> **⚠ HISTORICAL snapshot from 04-21 morning. For current state read
> `L49_HANDOFF.md`.** The "Recommended next experiments" at the bottom of this
> file have been superseded: we pivoted to per-query bad-expert suppression
> (L49) after discussing novelty with the user. The morning's recommendations
> 1-4 are not the current track.

## What I ran overnight

| Job    | What                                              | Status             |
|--------|---------------------------------------------------|--------------------|
| L46    | weight-ortho (all 1024 experts) jailbreak eval    | DONE (4h 07m)      |
| L47d   | 25-attack causal Δ fingerprint extraction         | DONE               |
| L47e   | hierarchical clustering + refusal-engagement score| DONE               |
| L47f   | Δ → cluster classifier                            | DONE, **94.0% acc**|
| L47g   | single-expert (L=8, E=29) suppression defense     | partial (timed out)|
| L47g2  | focused bias=-20 followup (incremental CSV)       | DONE               |
| L47h   | multi-expert suppress / amplify / combined        | finishing          |

## Top-line findings

### 1. L47f: attack-family fingerprinting works (94.0% test acc)
- Clean 5-cluster structure in 25-attack causal Δ space (k=5 chosen by silhouette).
- Classifier maps a **single query's 1024-dim Δ** to cluster label at 94.0% acc.
- 17 of 25 attacks are "bypass" (refusal-engagement score <0.3); 6 are
  "engage-but-fail" (score >0.5). Different mechanistic failure modes.

### 2. L46 weight-ortho: marginal, not paper-carrying
- ortho_all (remove d_refuse from all 1024 experts' W_down) gives:
  - `none` safe%: 48 → **54** (+6)
  - `new_gpt4_cipher`: 8 → 12 (+4)
  - `new_pair`: 25 → 25 (0)
  - `past_tense`: 32 → 29 (-3)
  - benign: 100% (no helpfulness damage)
- **P1 story (weight-level immunization) is no longer compelling.**

### 3. L47g/g2/h: router-level defense fails on bypass attacks
- Single-expert neg bias on E29 (the top correlate for bypass clusters): **zero effect on bypass refusal rates**.
- Multi-expert (5-expert) neg bias on candidate "suppressor" experts: also zero effect on bypass.
- Amplify (pos bias on cluster-5 "refusal driver" experts): destroys coherence (pure gibberish).
- **Routing is redundant — no single or small group of experts monopolizes the suppression function.**

### 4. Positive side finding: L47h suppress on ENGAGED attacks
- Same 5-expert suppression boosts `none` (60 → 67%) and **GCG (20 → 47%)** refusal.
- Suppress-only condition is the cleanest mechanistic lever we've found.
- Caveat: keyword-based refusal detection is slightly conservative; true rates may be higher still.

## Recommended paper direction (P2+ with pivot)

**P1 (weight-ortho immunization) is dead**; **P2 (mechanistic taxonomy) is stronger than expected**.

Proposed title: _"Mechanistic taxonomy of MoE jailbreak attacks: attack-family fingerprints without a local defense"_

Three clean contributions:
1. **Method.** Per-query causal Δ extraction via forward-gradient on router-bias — produces a 1024-dim fingerprint of which experts drive / suppress refusal per query.
2. **Finding.** 25 jailbreak attacks cluster into 5 mechanistic families; a linear classifier on Δ predicts family at 94% accuracy from a single query.
3. **Negative result (surprising and useful).** Three structural interventions — weight orthogonalization, single-expert router suppression, multi-expert router suppression — all fail to restore refusal on bypass-cluster attacks. Routing redundancy limits expert-level defense.

Side contribution: suppression of candidate "suppressor" experts ~doubles refusal on engage-but-fail attacks (GCG 20→47%) without hurting benign.

## Recommended next experiments (for you to greenlight)

Priority order if you want to solidify P2 for publication:

1. **Replicate on a second MoE** (Qwen1.5-MoE-A2.7B or DeepSeek-V2-Lite).
   If the same 5-cluster structure + 94% classifier recur on another MoE, the taxonomy is robust.
2. **Re-score L47h with Qwen judge** (not keyword regex). The 94% classifier + suppress-engaged-attack finding deserve rigorous scoring.
3. **Zero-ablation of E29** (hard intervention: set expert W_down output to zero, not just router bias). If THIS also fails on bypass, the negative-result claim is airtight.
4. **Ablation of Δ dimensionality** — how many (layer, expert) dims does the 94% classifier actually need? If top-50 suffices, that's paper-ready as a lightweight detector.

## Files worth reading

- `PIVOT_MEMO_L47.md` — full narrative with all data tables
- `/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em/olmoe/stage_l46_weight_ortho_summary.txt`
- `/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em/olmoe/stage_l47e_clusters.txt`
- `/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em/olmoe/stage_l47f_classifier.txt`
- `/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em/olmoe/stage_l47g2_bias_m20.csv`
- `/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em/olmoe/stage_l47h_multi_expert.csv`
