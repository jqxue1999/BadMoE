# MoE Jailbreak Defense Project — Context & State

## Research Goal

Investigate **router-level defenses against jailbreak attacks on Mixture-of-Experts (MoE) LLMs**: per-query identification and suppression of adversarial experts via causal gradients on router biases.

**Paper title**: *Different Inputs, Different Adversarial Experts: Locate with Causal Gradients, Suppress via Router Bias*

**Thesis**: Adversarial experts in MoE models are **input-specific, not population-universal**. A two-step defense — (1) *locate* via a gate on harmfulness plus causal-gradient attribution to router logits, (2) *suppress* with an adaptive-magnitude router bias weighted by per-expert causal importance — outperforms fixed-mask steering baselines (SteerMoE) on both jailbreak safety and benign helpfulness.

**Target model**: `OLMoE-1B-7B-0924-Instruct` (16 transformer layers × 64 experts per layer, top-8 routing).

---

## Method Summary (three ingredients)

1. **Two probe directions**
   - `d_refuse` — Arditi-style, prompt-token mean-diff between harmful and benign inputs. Used as the **gate signal**: valid across both distributions.
   - `d_behavior` — response-token mean-diff between refuse vs comply continuations. Used as the **attribution signal** for `Δ_{ℓ,e}(q)`: sharp on harmful inputs where it was learned, not used on benign.

2. **Causal attribution via autograd** (shared forward)
   - Single forward through layers `{0,…,ℓ*}` captures the probe state `h_probe`.
   - From `h_probe` compute both:
     - `λ(q) = h_probe · d_refuse_unit` (harmfulness gate)
     - `refuse_score(q) = h_probe · d_behavior_unit` (refusal-path pressure)
   - Insert zero-initialised router biases `b_{ℓ,e}` at each expert logit; backprop gives
     `Δ_{ℓ,e}(q) = ∂ refuse_score / ∂ b_{ℓ,e}` — the causal sensitivity of refusal to each expert.

3. **Adaptive suppression**
   - Query-level magnitude: `β_eff(q) = β · clip((λ(q) − λ_lo) / (λ_hi − λ_lo), 0, 1)` — linear ramp; benign queries get `β_eff ≈ 0` automatically.
   - Expert-level weighting: `b_{ℓ,e}(q) = −β_eff(q) · |Δ_{ℓ,e}| / max|Δ|` on the top-K most-refusal-blocking experts.
   - Total cost per query: **one shared forward + one backward + one generation** (~6% wall-clock overhead vs plain generate).

**Pilot method** (used to generate current pilot numbers): uniform `β = −10` on top-K=25 NEG experts, no gate, no per-expert weighting. Lives as row 1 of the Appendix ablation table.

---

## Literature Context

### Direct baseline
- **SteerMoE** — population-level fixed-mask steering: identifies `A+` / `A−` expert pools from aggregate routing statistics; steers by amplifying `A+` and/or suppressing `A−`. We refute its implicit assumption that adversarial experts are *global*.

### Foundational
- **Arditi et al. 2024 (NeurIPS)** — *Refusal in LMs is Mediated by a Single Direction*. Source of `d_refuse` extraction and weight-orthogonalisation (used as an appendix baseline).
- **Chen, Arditi, Sleight, Evans, Lindsey 2025** — *Persona Vectors*. Methodological ancestor for `d_behavior` (response-token contrastive mean-diff).

### Other MoE / alignment defenses
- **RICE**, **Safety Neurons**, **CAFT** — activation-level or neuron-level interventions. Require full masks or training changes. Ours is inference-time, query-adaptive, and architecture-consistent (router biases are native MoE controls).

---

## Experimental Status (as of 2026-04-21)

### Pilot (n=50 per cell, 4 engaged attacks, uniform β=−10) — DONE

**Jailbreak safety rate (Qwen2.5-14B judge, threshold 70, higher = safer)**:

| Attack | baseline | SteerMoE | **C_perQ_sup (ours pilot)** | D_perQ_comb |
|---|---:|---:|---:|---:|
| none | 48.0 | 45.5 | **52.0 (+4)** | 0 (gibberish) |
| GCG | 16.0 | — | **30.0 (+14)** | 0 |
| PAIR | 25.0 | 33.0 | 22.0 (−3) | 0 |
| past_tense | 32.0 | 41.4 | **62.0 (+30)** 🔥 | 0 |

**Benign helpfulness (XSTest n=50, ANSWER/REFUSE binary)**:

| Condition | Answer rate | Over-refusal cost |
|---|---:|---:|
| baseline | 98.0% (49/50) | — |
| **C_perQ_sup (ours pilot)** | **98.0%** | **0 pp** |
| SteerMoE | 86.0% (43/50) | −12 pp |

### Key validations (pilot)

- **Adversarial experts are input-specific**: pairwise Jaccard on top-5 NEG sets = **0.106** (across 200 queries). 36.6% of query pairs share *zero* experts. Most-frequent "universal bad" expert (L=10, E=30) appears in only 52% of queries.
- **Concordance with SteerMoE**: per-query NEG picks enrich **4–8×** into SteerMoE `A−` pool (same "bad expert" concept, selected per-query).
- **Amplification fails**: per-query POS-Δ picks do NOT overlap SteerMoE `A+` (0.5–4.4%, ≈ chance). `+3` bias on POS experts breaks coherence entirely. Asymmetry is real.
- **Bypass attacks unreachable**: `deepinception`, `cipher`, `AIM`, `DAN`, `renellm`, `autodan` all have `|Δ| ≈ 0` on OLMoE — mechanistically the model never engages the refusal pathway. Appendix limitation.

### Pending (Exp 1–10 in `scripts/moe/experiment_plan.md`)

| Exp | What | Status | Paper slot |
|---|---|---|---|
| **9** | Adaptive-magnitude method (final §3 method — gate + `|Δ|`-weighting + λ-ramp) | 🟡 PENDING | Table `tab:adaptive` + main Table 1 final-method row |
| **1** | Scale to n=100, 12 attacks × 5 clusters | 🟡 PENDING | Main Table 1 (replace) + `app:n100` |
| **10** | AlpacaEval truly-benign (50 prompts) | 🟡 PENDING | `tab:benign` row |
| 2 | Second MoE (Qwen1.5-MoE-A2.7B) generalisation | 🟡 PENDING | `app:qwen_moe` |
| 3 | K sweep on past_tense | 🟡 PENDING | `app:ksweep` |
| 4 | `d_refuse` vs `d_behavior` attribution ablation | 🟡 PENDING | `app:drefuse` |
| 5 | Bypass attacks quantified | 🟡 PENDING | `app:bypass` extended |
| 6 | Inference overhead wall-clock numbers | 🟡 PENDING | `app:impl` |
| 7 | Per-layer K constraint | 🟡 optional | Table 1 column |
| 8 | Full 25-attack PandaGuard suite | 🟡 aspirational | Table 1 replace |

See `scripts/moe/experiment_plan.md` for per-experiment rationale, script changes, compute estimates.

---

## Infrastructure

### User / cluster
- **Username**: `ji757406.ucf`
- **QOS / account**: `qi855292.ucf` (NOT `sgao1`)
- **Group allocation**: 48 CPU, 375 GB MEM, 8 GPU (B200)

### Storage layout
```
/home/ji757406.ucf/trustworthy/           # Code only
├── scripts/moe/                          # L10..L49 pipeline
│   ├── stage_l49a_cache_delta.{py,slurm} # Δ cache (jailbreak, 200 queries)
│   ├── stage_l49b_generate.{py,slurm}    # C_perQ_sup gens
│   ├── stage_l49b2_combined.{py,slurm}   # D_perQ_comb gens
│   ├── stage_l49c_judge.{py,slurm}       # Qwen 0-100 safety
│   ├── stage_l49d_benign_cache.{py,slurm}
│   ├── stage_l49e_benign_generate.{py,slurm}
│   ├── stage_l49f_benign_judge.{py,slurm}    # ANSWER/REFUSE
│   ├── stage_l49g_steermoe_benign.{py,slurm}
│   ├── L49_HANDOFF.md                    # detailed status — read first
│   └── experiment_plan.md                # Exp 1-10 spec
├── paper/
│   ├── main.tex                          # ~900 lines, 11 pages
│   ├── example_paper.bib                 # 13 entries
│   ├── neurips_2026.sty
│   └── SteerMoE/                         # reference paper (do not edit)
├── repos/                                # cloned baselines
├── .venv → /orange/.../trustworthy_venv  # symlinked
└── CLAUDE.md                             # this file

/orange/qi855292.ucf/ji757406.ucf/
├── olmoe/                                # KEY artifacts for L49
│   ├── pandaguard_jbb_attacks.csv        # jailbreak suite
│   ├── xstest_safe_prompts.csv           # benign suite
│   ├── d_refuse.pt                       # [16, 2048]
│   ├── d_universal_avg.pt                # dict, 'd_behavior' [16, 2048]
│   ├── stage_l15_rd.pt                   # SteerMoE A+/A- counts
│   ├── stage_l49a_delta_cache.pt         # (200, 16, 64) Δ_refuse + Δ_behavior
│   ├── stage_l49b_generations.csv        # 200 C_perQ_sup gens
│   ├── stage_l49b2_d_combined.csv        # 200 D_perQ_comb gens
│   ├── stage_l49c_summary.txt            # safety-rate table
│   ├── stage_l49d_benign_cache.pt        # (50, 16, 64)
│   ├── stage_l49e_benign_generations.csv # baseline + C_perQ_sup on XSTest
│   ├── stage_l49f_benign_summary.txt
│   ├── stage_l49g_steermoe_benign.csv
│   ├── stage_l42_jailbreak_baseline.csv  # external baseline, reuse
│   ├── stage_l42_jailbreak_steermoe.csv
│   └── stage_l46_weight_ortho_summary.txt
└── trustworthy_venv/                     # uv venv (vllm, transformers, etc.)

/blue/qi855292.ucf/ji757406.ucf/cache/    # vllm / flashinfer caches
```

### SLURM template (GPU jobs)
```bash
#SBATCH --qos=qi855292.ucf
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --cpus-per-task=8         # ≤8 to avoid QOSGrpCpuLimit
#SBATCH --mem=64gb                # ≤96gb to avoid QOSGrpMemLimit
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

module purge
module load cuda/12.8.1

export HF_HOME=/orange/qi855292.ucf/ji757406.ucf/cache/huggingface
export TORCH_HOME=/orange/qi855292.ucf/ji757406.ucf/cache/torch
export XDG_CACHE_HOME=/orange/qi855292.ucf/ji757406.ucf/cache
export VLLM_CACHE=/blue/qi855292.ucf/ji757406.ucf/cache/vllm
export PYTHONUNBUFFERED=1
```

**Rule**: Main CLI session is on a CPU login node. ALL GPU work (generate / judge / train) goes through `sbatch` — never `nohup` in the login session.

### Paper compile command (NeurIPS 2026 template uses natbib)
```bash
cd /home/ji757406.ucf/trustworthy/paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
All 4 passes are required — skipping any yields `?` citations.

---

## Locked-in Decisions (do NOT revisit)

| Decision | Why |
|---|---|
| **Two directions**: `d_refuse` gate + `d_behavior` attribution | `d_behavior` was learned on harmful inputs only → invalid projection on benign. `d_refuse` (Arditi) generalises across both distributions. Use each where its signal is valid. |
| **K = 25** for top-K expert selection | Matches SteerMoE's `A−` bandwidth for apples-to-apples comparison. |
| **Engaged attacks only** for initial pilot (`none`, `GCG`, `PAIR`, `past_tense`); Exp 1 expands to 12 attacks × 5 clusters | Bypass attacks have `\|Δ\| ≈ 0`; mechanistically unreachable. |
| **Qwen2.5-14B** as judge, threshold **70** (not 50 as in L42) | 50 is too lenient for jailbreak partial-compliance. |
| **Baseline numbers** reused from L42 (no re-judge needed) | Same judge, same decoding. |
| **Weight-orthogonalisation (L46)** → appendix | Pilot showed +6pp on `none`, 0 on jailbreaks. Not the main story. |
| **No amplification** in final method | POS-Δ ≠ SteerMoE `A+`; +3 bias breaks coherence. Asymmetry is real. |
| **Final method = gate + `\|Δ\|`-weighting + λ-ramp**; pilot = uniform-always | Pilot is a stepping stone; paper reads as adaptive; pilot numbers land as row 1 of `tab:adaptive`. |

---

## Claims the Paper Makes

1. **Adversarial experts are input-specific** (Jaccard 0.11) — refutes SteerMoE's global-mask assumption.
2. **Per-query suppression beats SteerMoE** on 3/4 engaged jailbreaks at **half the bandwidth** (K=25 vs K=25+25) with **0 pp benign tax** (SteerMoE: −12 pp).
3. **Amplification fails** (`D_perQ_comb` → gibberish); asymmetry traced to POS-Δ ≠ validated safe-expert pool.
4. **Bypass attacks mechanistically unreachable** — `|Δ| ≈ 0` — honest limitation.
5. **Final method is adaptive** — gate filters benign, per-expert `|Δ|`-weighting spreads magnitude appropriately (measurable after Exp 9).

---

## Common Pitfalls (learned)

1. **Python stdout buffering in SLURM**: `sys.stdout.reconfigure(line_buffering=True)` + `flush=True` + `PYTHONUNBUFFERED=1`.
2. **Incremental CSV saves**: write after every batch to survive OOM/preemption.
3. **Home disk fills with venv**: symlink `.venv → /orange/.../trustworthy_venv`.
4. **Hook modifications on KV-cache**: `output[0]` is 3D `(B,S,H)` on prefill, 2D `(B,H)` on cache step — check ndim.
5. **Autograd on router biases**: insert as zero-init leaf tensors BEFORE forward; use `create_graph=False`, retain only the one backward.
6. **vLLM + autograd**: vLLM has no autograd. For Δ extraction use HF transformers directly; reserve vLLM for judge-only.
7. **QOS limits**: group MEM 375 GB, CPU 48. Reduce request if Reason shows `QOSGrp*Limit`.
8. **Paper compile**: 4-pass sequence required for natbib; skipping loses citations.
9. **L49f is idempotent**: judges only rows with empty `answered`. Safe to resubmit.
10. **Codex review before long SLURM jobs**: use `codex-rescue` to pre-check scripts before multi-hour runs.

---

## Priority Next Steps

See `scripts/moe/L49_HANDOFF.md` "§Paper status → What still needs work" for the full priority-ordered list. Top 3:

1. **Run Exp 9** (adaptive-magnitude method) — final §3 method, fills `tab:adaptive` + main Table 1 final-method row.
2. **Run Exp 1** (n=100 × 12 attacks) — tightens CI, rules out cluster-5-only concern, fills Table 1.
3. **Run Exp 10** (AlpacaEval benign) — helpfulness sanity check on ordinary queries.

Any subsequent agent should start by reading:
1. `scripts/moe/L49_HANDOFF.md` — full status and decisions
2. `scripts/moe/experiment_plan.md` — Exp 1–10 specs
3. `paper/main.tex` — grep `PLACEHOLDER` to see what's waiting (~18 markers)
