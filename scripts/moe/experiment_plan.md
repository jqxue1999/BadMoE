# L49 Post-Submission-Draft Experiment Plan

**Purpose**: Everything that would strengthen the current draft before
submission, in priority order. Each section has (a) rationale, (b)
script to write/modify, (c) expected compute, (d) which paper
table/section the result feeds into. Placeholder data is currently in
the paper — replace with real data as each experiment finishes.

Reference paper: `paper/main.tex`. Tables already scaffolded with
`\textbf{[PLACEHOLDER]}` markers for every pending result.

---

## Tier 1 — must do before submission

### Exp 1: Scale jailbreak n=50 → n=100 AND expand attack suite

**Two motivations combined**:
1. **Statistical power**: 95 % binomial CI at $n{=}50$ is $\pm 14$ pp.
   The $+30$ pp past-tense result is bulletproof, but $+4$ pp (none),
   $+14$ pp (GCG), and $-3$ pp (PAIR) differences are inside the CI.
   $n{=}100$ halves CI to $\pm 10$ pp.
2. **Attack coverage**: Current Table 1 has 4 attacks, 3/4 in
   cluster 5. PandaGuard JBB has 25 attacks across 5 mechanistic
   clusters (per L47e). To rule out "works only on cluster-5 attacks",
   pick 2 attacks per cluster = 10 attacks.

**Target 10 attacks** (chosen as 2 per cluster):

| Cluster | Attacks |
|---|---|
| 5 engaged | `none`, `GCG`, `PAIR`, `ICA` (4; we already have 3) |
| 4 tense-shift | `past_tense`, `tense_future` |
| 3 DAN/roleplay | `AIM`, `BETTER_DAN` |
| 1 persuasion/search | `new_pair`, `new_renellm` |
| 2 pure bypass | `new_deepinception`, `new_gpt4_cipher` *(negative controls)* |

That's 12 attacks total. The bypass two serve as negative controls
confirming mechanistic unreachability, not as targets for improvement.
The 10 non-bypass attacks are where we expect wins.

**Also fixes SteerMoE GCG gap**: L42 SteerMoE reproduction only ran on
4 attacks (`none`, cipher, PAIR, past_tense), omitting GCG. Rerun here
fills the `—` cell in current Table 1.

**What to do**:
1. Edit `scripts/moe/stage_l49a_cache_delta.py`:
   - `ATTACKS = [list above]`
   - `N_PER_ATTACK = 100`
   - Output `stage_l49a_delta_cache_v2.pt`
2. Edit `stage_l49b_generate.py` + `stage_l49b2_combined.py` to read new
   cache.
3. **Rerun SteerMoE** on same 12 attacks at n=100:
   copy `stage_l49g_steermoe_benign.py` → `stage_l49g2_steermoe_jailbreak.py`
   and swap benign prompts for jailbreak prompts.
4. **Rerun baseline** on 8 new attacks (4 original already in L42;
   adapt `stage_l42_jailbreak_baseline.py` to run these extra attacks
   at n=100).
5. Judge all with `stage_l49c_judge.py` (idempotent across CSVs).

**Compute**:
- Δ cache: 12 × 100 = 1200 queries × 0.6 s = ~15 min
- Baseline gen: 8 new attacks × 100 × 20 s = ~4.5 h
  (4 original reuse L42, just pull its n=50 — or re-run to n=100 too for +2 h)
- C_perQ_sup gen: 12 × 100 × 20 s = ~6.7 h
- SteerMoE gen: 12 × 100 × 20 s = ~6.7 h
- D_perQ_comb gen: 12 × 100 × 20 s = ~6.7 h
  (can skip if we're time-pressed; already 0 at n=50 for 4 attacks)
- Judge: ~2 h

**Total: ~27 h B200** if all conditions + n=100 everywhere. Cheaper path
skip D_perQ_comb + reuse L42 n=50 baseline = **~18 h**.
Feasible over 2 days wall-clock.

**Paper slot**: Replace Table 1 entirely. New structure:

```
        Cluster  baseline  SteerMoE  C_perQ_sup  D_perQ_comb
none         5     ...       ...       ...         0
GCG          5     ...       ...       ...         0
PAIR         5     ...       ...       ...         0
ICA          5     ...       ...       ...         0
past_tense   4     ...       ...       ...         0
tense_future 4     ...       ...       ...         0
AIM          3     ...       ...       ...         0
BETTER_DAN   3     ...       ...       ...         0
new_pair     1     ...       ...       ...         0
new_renellm  1     ...       ...       ...         0
---- bypass (negative control) ----
new_deepinception  2  ...  ...  ...  0
new_gpt4_cipher    2  ...  ...  ...  0
```

Also revise §Experiments prose to cite cluster structure as coverage
argument.

---

### Exp 2: Second MoE architecture (generalisation check)

**Rationale**: The single most frequent reviewer critique. Currently
everything is on OLMoE-1B-7B. If the 94 % classifier (L47f), per-query
diversity (Jaccard 0.11), and suppression gains reproduce on a second
MoE, we go from "OLMoE-specific curiosity" to "MoE-general method."

**Candidate models** (in preferred order):
1. **Qwen1.5-MoE-A2.7B** — 60 experts × 24 layers, close sizing to
   OLMoE, Apache-2.0 license. Recommended.
2. **DeepSeek-V2-Lite** — 64 experts × 27 layers, similar but more
   experts. License has commercial restrictions; check.
3. **Mixtral-8x7B** — only 8 experts, so per-query top-K is less
   meaningful. Skip unless others fail.

**What to do**: Port the entire L49 pipeline to the new model. The
key changes:
1. `stage_l15_rd.pt` equivalent — re-run SteerMoE reproduction on
   the new MoE (follow L40 blueprint).
2. `d_behavior.pt` equivalent — re-run L45c/L45e to extract the
   behavioral direction for the new MoE.
3. `stage_l49a_cache_delta.py` — adjust `MODEL_ID`, `n_layers`,
   `n_experts`, `PROBE_LAYER` (likely ~40 % through the model).
4. Same for `stage_l49b_*`, `stage_l49c_*`, `stage_l49d_*`, `stage_l49e_*`,
   `stage_l49f_*`, `stage_l49g_*`.
5. Re-run full pipeline on 4 engaged attacks + XSTest.

**Compute**: ~8–12 h B200 (including SteerMoE reproduction). Spread
over 2 days realistically.

**Paper slot**: Appendix table `tab:qwen_moe` (currently placeholder).
Update §5 Discussion with a sentence on generalisation.

---

### Exp 3: K sweep ablation

**Rationale**: We picked $K{=}25$ to match SteerMoE's bandwidth, but
didn't sweep. Reviewer will ask "is $K$ optimal?"

**What to do**:
1. Copy `scripts/moe/stage_l49b_generate.py` → `stage_l49h_ksweep.py`.
2. Add outer loop `for K in [5, 10, 25, 50, 100]:`.
3. Run on **past-tense only** (our strongest signal, cheapest to test).
   Single condition: $C_{\text{perQ-sup}}$.

**Compute**: 5 × 50 gens × 20 s = **~1 h B200** + 10 min judge.

**Paper slot**: Appendix Figure `fig:ksweep` — line plot of safety rate
vs $K$ on past-tense. Expected shape: plateau around $K{=}10$–$50$,
sharp drop at $K{=}5$ (too sparse) or $K{=}100$ (over-intervention).

---

## Tier 2 — strongly recommended

### Exp 4: d_refuse per-query ablation

**Rationale**: Anonymous reviewer will ask "is d_behavior genuinely
better than d_refuse, or does it not matter?" We have the cached
`delta_refuse` in `stage_l49a_delta_cache.pt` — testing it is one-line
code change.

**What to do**:
1. Copy `stage_l49b_generate.py` → `stage_l49i_drefuse.py`.
2. One-line change: `topk_neg_experts(delta_b[qi], ...)` →
   `topk_neg_experts(delta_r[qi], ...)` (swap the field name).
3. Rename condition → `C_perQ_refuse` to distinguish from
   `C_perQ_behavior`.
4. Run same 200 queries (n=50 per attack).
5. Judge with L49c (multi-CSV aware).

**Compute**: 200 gens × 20 s = **~1 h B200** + 10 min judge.

**Paper slot**: New Table 2' in §4.2 — "$C_{\text{perQ-behavior}}$ vs
$C_{\text{perQ-refuse}}$". Expected: d_behavior wins by small but
consistent margin.

---

### Exp 5: Bypass attacks quantified limitation

**Rationale**: Paper says bypass attacks are "mechanically unreachable."
Reviewer asks "data, please." Make it quantitative.

**What to do**:
1. Copy `stage_l49a_cache_delta.py` → `stage_l49j_bypass_cache.py`;
   change `ATTACKS` from `[none, GCG, PAIR, past_tense]` to
   `[new_deepinception, new_gpt4_cipher, AIM]` (three representative
   bypass attacks).
2. Copy `stage_l49b_generate.py` → `stage_l49k_bypass_gen.py`; point
   at the new cache.
3. Run condition $C_{\text{perQ-sup}}$ on bypass queries.
4. Add rows to Qwen judge.

**Compute**: cache ~5 min + 3 × 50 gens × 20 s = **~50 min** + 15 min
judge.

**Paper slot**: Extend Appendix `app:bypass` table with
`C_{\text{perQ-sup}}` column.

Expected result: safety rate $\approx$ baseline (no improvement),
confirming mechanistic unreachability.

---

## Tier 1 addendum — adaptive-magnitude method (now part of §3)

### Exp 9: Adaptive-magnitude defense (gate + expert weighting)

**Rationale**: Method-defining, not ablation. Current paper §3.4 specifies
$\beta_{\text{eff}}(q) = \beta \cdot f(\lambda(q))$ with linear ramp +
$|\Delta|$-weighting per expert. Needs (i) calibration of
$\lambda_{\text{lo}}, \lambda_{\text{hi}}$, (ii) implementation, (iii)
ablation table at Appendix~\ref{app:adaptive}.

**What to do (in order)**:

1. **Augment L49a cache** — re-run with extra column for $\lambda(q) =
   \mathbf{h}^*_{\text{probe}} \cdot \drefuse$ saved alongside $\Delta$.
   No extra autograd, just one dot product per query.
   Edit: `scripts/moe/stage_l49a_cache_delta.py` to additionally capture
   `h_probe` and compute `lambda_q = h_probe @ d_refuse_unit`. Save to
   cache dict.

2. **Calibrate $\lambda$ thresholds** — one-off script
   `stage_l49n_calibrate_lambda.py`:
   - Load cache over the 12 attack × 100 prompts + 50 XSTest benign +
     50 AlpacaEval (see Exp 10).
   - Compute $\lambda$ distribution per class.
   - Set $\lambda_{\text{lo}} = $ 95th percentile on benign pool.
   - Set $\lambda_{\text{hi}} = $ 5th percentile on harmful pool.
   - Save to `stage_l49n_thresholds.pt`.

3. **Four variants for ablation** — copy `stage_l49b_generate.py` →
   `stage_l49o_adaptive.py` with conditions:
   - `uniform_always`   : current $C_{\text{perQ-sup}}$ (K=25, $\beta=-10$, no gate)
   - `hard_gate`        : $\mathbf{1}[\lambda > \tau]$ then uniform $-10$
   - `ramp_uniform`     : $\beta_{\text{eff}}(q)$ ramp, all $K$ experts same magnitude
   - `ramp_weighted`    : $\beta_{\text{eff}}(q)$ ramp + $|\Delta|$-weighting (the final method)
4. **Evaluate on 3 distributions**:
   - engaged jailbreak (12 attacks × 50, from Exp 1 cache)
   - XSTest benign (50, reuse L49e cache)
   - AlpacaEval benign (50, new; see Exp 10)
5. Judge all with existing L49c (jailbreak) + L49f (benign) logic.

**Compute**: 4 conditions × (12 × 50 + 50 + 50) = 2800 gens × 20 s =
$\sim$ 15 h B200. If tight, skip `ramp_uniform` (keep 3 variants) for
$\sim$ 11 h.

**Paper slot**: Table `tab:adaptive` in Appendix \ref{app:adaptive}
(already scaffolded with PLACEHOLDER). The \emph{ramp\_weighted} row is
the ``our final method'' row and should also feed the main Table~\ref{tab:main}
once numbers are in.

---

### Exp 10: Truly benign benchmark (AlpacaEval)

**Rationale**: L49f showed 0 pp over-refusal on XSTest, but XSTest is
adversarial-benign (harmful-looking). Need a benchmark of ordinary
helpful requests (``Write a Python function to reverse a string'',
``What's the capital of France?''). AlpacaEval 2.0 or a random
$n{=}50$ subset is standard.

**What to do**:
1. Download AlpacaEval 2 prompts; take first 50 (or stratified
   sample).
2. Run through $C_{\text{perQ-sup}}$ at $K{=}25$, $\beta{=}-10$
   (current), then re-run with the adaptive scheme from Exp 9.
3. Judge with L49f ANSWER/REFUSE classifier.

**Compute**: 2 × 50 gens × 20 s + judge $\sim$ 1 h.

**Paper slot**: New Table~\ref{tab:benign} row (or side table)
reporting answer rate on AlpacaEval alongside XSTest. Key claim:
adaptive scheme preserves AlpacaEval answer rate within 1 pp of
baseline.

---

## Tier 3 — if time permits

### Exp 6: Inference overhead wall-clock

**Rationale**: We say "~0.6 s for Δ extraction." Measure it.

**What to do**:
1. Small benchmark script that times `Delta_extract` + `generate_with_hook`
   over 10 queries, compare to plain `generate`.
2. Report mean + stddev.

**Compute**: 5 min.

**Paper slot**: §Appendix `app:impl` — one sentence with numbers.

---

### Exp 7: Per-layer K constraint

**Rationale**: Current top-25 is global flat. May cluster in a few
layers. Constrained version might be gentler.

**What to do**:
1. Copy `stage_l49b_generate.py` → `stage_l49m_layerk.py`.
2. Change `topk_signed_experts` to enforce at most 3 experts per layer.
3. Run on all 4 attacks n=50.

**Compute**: 200 gens × 20 s = ~1 h.

**Paper slot**: Add one column to Table 1 or put in Appendix.

---

### Exp 8: Full 25-attack PandaGuard suite (aspirational)

**Rationale**: Currently only 4 engaged attacks. A full suite avg
would be impressive.

**What to do**: Scale Exp 1 from 4 attacks to 25. Most will be bypass
(low gains, matching our claim).

**Compute**: 25 × 50 × 2 conditions × 20 s = **~14 h B200**. Deprioritised.

**Paper slot**: Replace Table 1 with 25-row table or move existing
to Appendix.

---

## Summary execution plan

If submission is ~1 week away:

- **Day 1**: Exp 9 (adaptive-magnitude defense — now part of main method).
  Augment cache + calibrate $\lambda$ thresholds + 4-variant ablation.
  ~15 h compute, likely splits across 2 days.
- **Day 2**: Exp 1 (n=100 + 12 attacks) + Exp 10 (AlpacaEval benign).
  ~18-27 h compute, overnight.
- **Day 3**: Exp 3 (K sweep) + Exp 4 (d_refuse ablation) + Exp 5 (bypass).
  ~3 h compute.
- **Day 4–5**: Exp 2 (Qwen1.5-MoE). ~8 h compute, 2 days wall.
- **Day 6**: Exp 6 (timing) + Exp 7 (per-layer K).
- **Day 7**: Paper writing: fill placeholder tables, write up
  discussion, add figures.

If submission is < 3 days away, prioritise: Exp 9 (adaptive — core
method) + Exp 1 (n=100) + Exp 10 (AlpacaEval). ~25 h compute. Call out
Exp 2 (second MoE) as ``future work.''
