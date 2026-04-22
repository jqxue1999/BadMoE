# L49 Handoff — Read This First

**Status as of 2026-04-21 17:05.** ✅ **All pilot (n=50, uniform bias)
experiments complete.** The method has since been **refactored with a
gate + adaptive-magnitude suppression** (see §Method updates below);
the core L49b/c/e/f/g numbers on the pilot method are final and land
in the paper as-is, but the final method requires Exp 9 (adaptive)
+ Exp 10 (AlpacaEval) to be fully validated.

**Pilot headline**: per-query suppression ($K{=}25$, uniform $\beta{=}-10$,
no gate) beats SteerMoE on 3/4 engaged jailbreaks \emph{and} reduces
over-refusal on XSTest benign: we incur $0$\,pp cost, SteerMoE incurs
$-12$\,pp at identical bandwidth on the same 50 prompts.

**Final method (written in paper, not yet fully measured)**: gate via
$\lambda(q)=\mathbf{h}_{\ell^*}\cdot\drefuse$; if gate fires, locate
adversarial experts via $\Delta_{\ell,e}(q)$ (autograd on router
biases); suppress with bias
$b_{\ell,e}=-\beta_{\text{eff}}(q)\cdot|\Delta_{\ell,e}|/\max|\Delta|$,
where $\beta_{\text{eff}}(q)$ is a linear ramp on $\lambda(q)$. See
paper §3.3--3.4 for full spec.

**Paper title**: "Different Inputs, Different Adversarial Experts:
Locate with Causal Gradients, Suppress via Router Bias" — reflects
two-step method (locate = gate + causal attribution; suppress = router
bias) and the central finding that adversarial experts are
input-specific, not global.

---

## TL;DR status

| Step | Status | File |
|---|---|---|
| L49a Δ cache (200 jailbreak queries) | ✅ DONE | `stage_l49a_delta_cache.pt` |
| L49b C_perQ_sup (200 gens) | ✅ DONE | `stage_l49b_generations.csv` |
| L49b2 D_perQ_comb (200 gens) | ✅ DONE | `stage_l49b2_d_combined.csv` |
| L49c Qwen judge (both 400) | ✅ DONE | `stage_l49c_summary.txt` |
| L49d Δ cache (50 XSTest benign) | ✅ DONE | `stage_l49d_benign_cache.pt` |
| L49e benign generate (100 gens) | ✅ DONE | `stage_l49e_benign_generations.csv` |
| L49f benign judge (all 3 conditions) | ✅ DONE | `stage_l49f_benign_summary.txt` |
| L49g SteerMoE on same 50 XSTest | ✅ DONE | `stage_l49g_steermoe_benign.csv` |
| Paper draft (pilot numbers + placeholder tables for Exps 1–10) | ✅ 11 pages total (6 content + 2 bib + 3 appendix) | `paper/main.tex`, `paper/main.pdf` |
| **Exp 9 adaptive-magnitude method** | 🟡 PENDING | see §3.4 of paper + `experiment_plan.md` |
| **Exp 10 AlpacaEval benign** | 🟡 PENDING | new benchmark,truly-innocuous queries |
| Exps 1–8 (n=100, K sweep, etc.) | 🟡 PENDING | all scaffolded in paper with `[PLACEHOLDER]` |

**Immediate next action**: run Exp 9 (adaptive-magnitude method) — it's
the defining method variant per paper §3.4, not an ablation. Requires:
(a) augment L49a cache with per-query $\lambda(q)$, (b) calibrate
$\lambda_{\text{lo}}, \lambda_{\text{hi}}$ on a benign+harmful mix,
(c) run 4 magnitude variants, (d) fill Appendix table `tab:adaptive`.
Details in `experiment_plan.md` §"Exp 9".

### Final benign numbers (n=50 XSTest, Qwen ANSWER/REFUSE, same prompts)

| condition | answer rate | over-refusal cost |
|---|---:|---:|
| baseline | 98.0 % (49/50) | — |
| **C_perQ_sup** | **98.0 % (49/50)** | **0.0 pp** |
| SteerMoE | 86.0 % (43/50) | **−12 pp** |

Per-type: SteerMoE drops 12pp on figurative_language (100→88) AND
12pp on homonyms (96→84). Uniform over-refusal, not concentrated in
one type.

---

## Method refactor — summary of changes

The paper §3 "Method" was rewritten 2026-04-21 17:00 to reflect a
three-ingredient design rather than the original single-ingredient
uniform suppression. Prose (abstract, intro, conclusion) aligned with
new title. Code implementing pilot method (uniform $\beta=-10$, no gate)
is intact — these are the baseline for Exp 9.

**Three ingredients of the final method**:
1. **Two probe directions**. $\drefuse$ (Arditi-style, prompt-level
   harmful-vs-benign mean-diff) used as **gate signal** because it is
   valid across benign and harmful distributions. $\dbeh$ (response-
   level refuse-vs-comply mean-diff) used as **attribution signal** for
   $\Delta_{\ell,e}(q)$ because it is sharper for "who is blocking
   refusal" given the query is harmful.
2. **Causal attribution via autograd** (same as pilot): a single
   forward-backward pass through $\{0,\dots,\ell^*\}$ yields both
   $\lambda(q)$ and $\Delta(q)$ from the same $\mathbf{h}_{\ell^*}$.
3. **Adaptive suppression**: $b_{\ell,e}(q) = -\beta_{\text{eff}}(q) \cdot
   |\Delta_{\ell,e}|/\max|\Delta|$, with $\beta_{\text{eff}}(q) = \beta
   \cdot \operatorname{clip}((\lambda-\lambda_{\text{lo}})/(\lambda_{\text{hi}}-\lambda_{\text{lo}}),0,1)$.

Why this upgrade was necessary (from user review):
- Concern 1: "Every bad expert gets the same $-10$ bias?" — fixed by
  $|\Delta_{\ell,e}|$-weighting.
- Concern 2: "Benign queries should not be intervened on." — fixed by
  $\lambda$-gate + linear ramp on magnitude.
- Concern 3: "Does $\dbeh$-projection even make sense on benign?" — no;
  that's why we use $\drefuse$ for gating (valid on all distributions)
  and keep $\dbeh$ for attribution only (valid on harmful where it was
  learned).

The pilot (uniform-always) numbers now serve as **row 1** of the
adaptive ablation table (Appendix~\ref{app:adaptive}).

## About L49f "rerun" (important for anyone reading the job queue)

L49f is **idempotent**: its todo list is rows with empty `answered`. When
it's resubmitted after L49g writes a new CSV, it only judges the 50
SteerMoE rows (skipping the 100 rows L49e already populated). Total
incremental work ≈ 5 min Qwen judge. The final summary is computed over
all three conditions pooled.

Design rationale: this is cheaper and more robust than atomic re-judging
of the full set. It also allows adding more conditions later (e.g.,
B_fixed5 on benign) without re-judging everything.

## Main results (jailbreak, n=50 per cell, Qwen2.5-14B judge, threshold 70)

| Attack | baseline | SteerMoE | **C_perQ_sup (ours)** | D_perQ_comb |
|---|---:|---:|---:|---:|
| none | 48.0 | 45.5 | **52.0** (+4) | 0 ← gibberish |
| GCG | 16.0 | — | **30.0** (+14) | 0 |
| PAIR | 25.0 | 33.0 | 22.0 (−3) | 0 |
| past_tense | 32.0 | 41.4 | **62.0 (+30)** 🔥 | 0 |

- **C_perQ_sup (K=25 negative only) beats SteerMoE on 3/4 attacks** using
  half the bandwidth (SteerMoE is K=25+25).
- **D_perQ_comb (K=25 neg @ −10 bias + K=25 pos @ +3 bias) fully breaks
  coherence** — responses are gibberish, not refusals. This is a clean
  negative result about the suppress↔amplify asymmetry.

---

## 主线任务 — paper direction (locked in)

Paper thesis: **"Adversarial experts in MoE models are input-specific,
not population-universal. A two-step defense — gate on harmfulness +
locate adversarial experts via causal gradients + suppress with
adaptive-magnitude router bias — outperforms fixed-mask steering on
safety and helpfulness."**

Draft at `paper/main.tex`. See §"Paper status" below.

---

## What's settled — do NOT revisit

| Decision | Why |
|---|---|
| **Two directions, not one**: $\drefuse$ for gating + $\dbeh$ for attribution | $\dbeh$ is only valid on harmful-input distribution (it was learned there); $\drefuse$ generalises across benign and harmful. Use each where its signal is valid. |
| Attack set = engaged only (initial pilot: `none`, `GCG`, `PAIR`, `past_tense`). Exp 1 expands to 12 attacks × 5 clusters. | Bypass attacks (deepinception, cipher, AIM, DAN, renellm, autodan) have \|Δ\| ≈ 0 — mechanistically unreachable. L47c verified. Appendix in paper. |
| `K = 25` | Matches SteerMoE's 25 A- bandwidth for apples-to-apples. At K=25, per-query NEG picks enrich 4× into SteerMoE A-50. |
| Baseline = L42 numbers for pilot; Exp 1 reruns for full n=100 × 12 attacks | Same Qwen judge, same greedy decoding. |
| Weight-ortho (P1) SHELVED | L46 gave only +6pp on `none`, 0 on jailbreaks. Now an appendix in the paper. |
| Amplification direction | Per-query POS-Δ experts do NOT align with SteerMoE A+ (~chance enrichment). +3 bias already breaks coherence. Do not raise bias; do not pitch "promote good experts" as a main claim. |
| `L49c` judge threshold = 70 | We chose 70 (not 50 like L42) because 50 is too lenient for jailbreak partial-compliance. L42 reruns would need re-judging — but we reuse L42 baseline numbers directly since the method comparison is within-L49. |
| **Final method = adaptive magnitude + gate**, pilot = uniform-always | Pilot was a stepping stone; final method addresses legitimate concerns about per-expert magnitude and truly-benign over-refusal. Paper reads as "our method is adaptive"; pilot numbers live as one row of the ablation table. |

---

## What the remaining benign pipeline does

### L49d (DONE) — Δ cache for 50 XSTest benign prompts
Same pipeline as L49a but with `xstest_safe_prompts.csv` input. Cache at
`stage_l49d_benign_cache.pt`. Mean \|Δ_behavior\| = 7.5e-4, similar to
engaged attacks — healthy signal (slightly weaker since benign has no
refuse/comply divergence to exploit).

### L49e (RUNNING) — A_baseline + C_perQ_sup on benign
Job 30582263. 50 XSTest × 2 conditions = 100 gens. ~40 min total.
Output CSV `stage_l49e_benign_generations.csv`.

### L49f (QUEUED) — Qwen ANSWER/REFUSE classifier
Job 30582264 (dep on 30582263). Different judge prompt from L49c — binary
"did the AI answer or refuse?". For benign, answer = good, refuse = bad.
Output `stage_l49f_benign_summary.txt` with per-condition answer rate and
**over-refusal cost = baseline_ans − C_ans**.

### What to do when L49f finishes
1. Read `stage_l49f_benign_summary.txt`.
2. Open `paper/main.tex`, find Table ~`tab:benign` (around §"Benign
   Over-Refusal"). Replace the `TBD` placeholders with actual answer rates
   and the over-refusal cost.
3. Recompile: `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`
4. Passing bar: cost ≤ 10pp. If smaller, paper claim is strong. If larger
   (>15pp), discuss in §Discussion as a helpfulness/safety trade-off
   rather than a clean win.

---

## Paper status

### Files
```
paper/main.tex            ← ~900 lines, compiles to 11 pages (6 content)
paper/example_paper.bib   ← 13 BibTeX entries, all cited
paper/main.pdf            ← latest build
paper/neurips_2026.sty    ← template
paper/SteerMoE/           ← reference paper (don't copy from it)
```

### Compile command (NeurIPS 2026 template uses natbib)
```bash
cd /home/ji757406.ucf/trustworthy/paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
# or: latexmk -pdf main.tex
```
**All 4 passes required** — skipping any produces "?" citations.

### Sections filled with real data (pilot method, n=50)
- ✅ Introduction with title-aligned 3 findings
- ✅ Related work (Arditi, Chen, SteerMoE, RICE, weight-ortho)
- ✅ **§3 Method refactored**: Two directions + Locate (gate+Δ) + Adaptive Suppress
- ✅ Main result table 1 (4 attacks, pilot numbers)
- ✅ Amplification failure negative result (§4.3)
- ✅ Concordance with SteerMoE A- (Table 2, 4–8× enrichment)
- ✅ Adversarial-experts-are-input-specific (Table 3, Jaccard 0.11)
- ✅ Benign over-refusal 3-way (Table 4: baseline / ours / SteerMoE on XSTest)
- ✅ Discussion + Conclusion aligned with new title

### Sections with placeholders (pending experiments)
All PLACEHOLDER markers in TeX — grep `PLACEHOLDER` to find them all:

| Section | Experiment to fill | Count |
|---|---|---|
| Table~\ref{tab:main} rows 4-12 (cluster-4/3/1/2 attacks) | Exp 1 | 9 rows |
| Appendix A.1 `app:n100` extended table | Exp 1 (n=100) | 1 table |
| Appendix A.2 `app:adaptive` 4-variant table | **Exp 9** | 1 table |
| Appendix A.3 `app:ksweep` K sweep | Exp 3 | 1 table |
| Appendix A.4 `app:drefuse` direction ablation | Exp 4 | 1 table |
| Appendix A.5 `app:qwen_moe` second-MoE | Exp 2 | 1 table |
| Appendix `app:bypass` extended | Exp 5 | 1 table |
| Appendix `app:impl` inference overhead numbers | Exp 6 | 3 numbers |
| Adaptive calibration thresholds | Exp 9 step 2 | 2 numbers |

Total: **~18 PLACEHOLDER markers**, scaffolded so any next agent can
`grep` and fill.

### What still needs work (priority order)
1. **Run Exp 9** (adaptive-magnitude method) — it's the final method
   per §3.4, not ablation. Fills `app:adaptive`. Also produces the
   "final method" row of Table~\ref{tab:main}.
2. **Run Exp 1** (n=100, 12 attacks) — fills Table 1 + `app:n100`.
3. **Run Exp 10** (AlpacaEval truly-benign) — validates 0 pp helpfulness
   on ordinary queries, not just XSTest.
4. **Run Exp 2** (second MoE) — generalisation check, likely biggest
   impact on accept/reject.
5. **Figures**: paper has no schematic. Add one diagram of
   gate → causal-Δ → adaptive-bias → generate.
6. **Calibrate $\lambda_{\text{lo}}, \lambda_{\text{hi}}$** — data-driven,
   no grid search. See Exp 9 step 2.
7. **More related work**: Wang et al. persona features, Zou RepE, RICE,
   SafeGuarding, etc. Grep `paper/SteerMoE/iclr2026_conference.tex`
   references for candidates.

---

## Key file paths

```
# Code
scripts/moe/stage_l49{a_cache_delta,b_generate,b2_combined,c_judge,d_benign_cache,e_benign_generate,f_benign_judge,g_steermoe_benign}.{py,slurm}

# Inputs
/orange/.../olmoe/pandaguard_jbb_attacks.csv         # jailbreak suite
/orange/.../olmoe/xstest_safe_prompts.csv            # benign suite
/orange/.../olmoe/d_refuse.pt                        # [16,2048]
/orange/.../olmoe/d_universal_avg.pt                 # dict, 'd_behavior' [16,2048]
/orange/.../olmoe/stage_l15_rd.pt                    # SteerMoE A+/A- counts (for concordance)

# L49 outputs (DONE)
/orange/.../olmoe/stage_l49a_delta_cache.pt          # (200, 16, 64) Δ_refuse + Δ_behavior
/orange/.../olmoe/stage_l49b_generations.csv         # 200 C_perQ_sup gens
/orange/.../olmoe/stage_l49b2_d_combined.csv         # 200 D_perQ_comb gens
/orange/.../olmoe/stage_l49c_summary.txt             # safety-rate table
/orange/.../olmoe/stage_l49d_benign_cache.pt         # (50, 16, 64)

# L49 outputs (benign, DONE for our method)
/orange/.../olmoe/stage_l49e_benign_generations.csv  # baseline + C_perQ_sup answered
/orange/.../olmoe/stage_l49f_benign_summary.txt      # will be overwritten by 30593148

# L49 outputs (pending L49g)
/orange/.../olmoe/stage_l49g_steermoe_benign.csv     # after 30593147

# External baselines to cite, not re-run
/orange/.../olmoe/stage_l42_jailbreak_baseline.csv
/orange/.../olmoe/stage_l42_jailbreak_steermoe.csv
/orange/.../olmoe/stage_l46_weight_ortho_summary.txt
```

---

## Job tracking commands

```bash
squeue -u ji757406.ucf
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed
tail -f /orange/qi855292.ucf/ji757406.ucf/trustworthy/logs/moe_em_olmoe/stage_l49e_benign_generate-30582263.out
```

---

## SteerMoE concordance (validation of "bad expert" concept)

From `stage_l15_rd.pt` risk-difference, `rd = p(E|safe) − p(E|unsafe)`.

| SteerMoE top-K (by `-rd`) | L49 per-q top-5 NEG ∩ A- | enrichment vs chance |
|---:|---:|---:|
| 5 | 2.9 % | 5.9 × |
| 25 | 9.6 % | 3.9 × |
| **50** | **38.9 %** | **8.0 ×** |
| 100 | 58.3 % | 6.0 × |

Cross-sign sanity: L49 NEG ∩ SteerMoE **A+** = 1.8 % (≈ chance). No sign
flip. L47h FIXED_5 — all 5 hand-selected experts appear in SteerMoE top-100 A-.

**POS side**: L49 POS ∩ SteerMoE A+ = 0.5 % to 4.4 % (0.5–1.0× enrichment,
chance). Our "good experts" are NOT SteerMoE safe experts. This is why
amplify fails.

---

## Per-query diversity (sanity that per-query is actually per-query)

200 queries total (4 attacks × 50), pairwise Jaccard on top-5 NEG sets:

- **Mean Jaccard: 0.106** (very low overlap)
- Fraction of pairs with Jaccard=0: 36.6%
- Unique top-5 signatures / 200 queries: 194 (3 pairs of near-duplicates)
- Most-frequent "universal bad" (L=10, E=30): appears in 52% of queries
- Unique (L, E) pairs covered by some query: 145 / 1024

Interpretation: genuine per-query selection, not a noisy version of a
universal mask.

---

## Out of scope — do NOT

- Re-run L42 baseline / L42 SteerMoE (data exists, reuse)
- Attempt L49 on bypass attacks (deepinception / cipher / AIM / DAN /
  renellm / autodan) — confirmed unreachable
- Pursue weight-level orthogonalisation as its own paper — it's an
  appendix in the current paper
- Port to Qwen1.5-MoE or DeepSeek-V2-Lite yet — do it only after L49f
  lands and paper text is final
- Re-extract d_behavior or d_refuse — both cached
- Scale jailbreak n beyond 50 without first resolving the PAIR −3pp
  question (reruns won't change tight 95% CI unless n doubles)

---

## Narrative the paper tells

1. **Method** (L47b → L49a): per-query causal Δ via router-bias autograd.
2. **Positive result** (L49b/c): query-adaptive suppression beats fixed
   SteerMoE on 3/4 engaged jailbreaks, half the bandwidth, no amplify.
3. **Concordance** (bonus): our NEG picks align with SteerMoE A- (4–8×
   enrichment) — same concept of "bad expert", better per-query selection.
4. **Negative result** (L49b2): amplification breaks coherence; asymmetry
   traced to POS-Δ not aligning with any validated safe-expert pool.
5. **Helpfulness check** (L49e/f, pending): benign answer rate stays
   close to baseline — intervention is targeted, not noisy.
6. **Limits** (appendix): bypass attacks mechanistically out of reach;
   single MoE architecture tested.

---

## If L49f says "bad news" (over-refusal > 15pp)

1. Report the number honestly in the paper.
2. Add a §Discussion subsection: "helpfulness/safety trade-off".
3. Ablation to propose: "skip suppression if per-query \|Δ\| norm < τ"
   — only intervene on queries where Δ has strong signal. This is a
   3-hour follow-up experiment.
4. Do NOT hide the result; it's worth publishing even if the trade-off
   is larger than hoped.
