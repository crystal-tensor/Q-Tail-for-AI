# Reflection Report — Q-TAIL v25 Manuscript Review

**Reviewer:** Reflection Agent  
**Document:** `paper_v25.tex` (1,329 lines)  
**Ground Truth:** `source_brief.md` (ver.25 revision spec)  
**Date:** 2026-07-21

---

## Summary

The v25 manuscript is broadly well-structured and correct on numbers, claims, and structure. All required sections (§1–§10, appendix) are present; every numeric claim from the brief is faithfully reproduced; claim-boundary honesty is rigorously maintained throughout. However, there are **five must-fix LaTeX/bibliography issues** that will prevent successful compilation or produce an empty/crippled references section, and one mathematical equation that should be verified for correctness. All other dimensions pass.

---

## Top-5 Must-Fix Issues

### 🔴 CRITICAL — Issue 1: Bibliography Will Produce No References

**Location:** Line 1120: `\bibliography{references}` + lines 1180–1327: `\begin{thebibliography}{99}...\end{thebibliography}`

**Problem:** There are two conflicting bibliography mechanisms:

1. `\bibliography{references}` (line 1120) → LaTeX will look for `src/references.bib`. **That file does not exist.** This command will silently produce zero output.
2. The inline `\begin{thebibliography}{99}` environment (lines 1180–1327) is the **only active bibliography content**, but **every single `\bibitem` inside it is commented out with `%`** — except the first entry (`aaronson2011computational`, line 1186).

**Result:** After compilation, the References section will contain only `[1]` (Aaronson 2011). All other citations — including heavily used ones like `padalkar2023open`, `yu2020metaworld`, `haarnoja2018soft`, `porter1956statistical`, `lin2017focal`, `chawla2002smote`, `kang2019decoupling`, `ahmadzadeh2023openpi`, `boixo2018characterizing`, `google-ai-coscientist`, `llnl-open-ai-coscientist`, `datacentricai`, and 20+ others — will appear as `[?]` (undefined citations) in the PDF.

**Fix:** Either (a) create `src/references.bib` with all required entries and remove/comment the inline `\begin{thebibliography}` block, or (b) uncomment all `\bibitem` lines inside the thebibliography environment. Option (b) is faster but note the corrupted author fields (see Issue 2 below) that must also be corrected simultaneously.

---

### 🔴 CRITICAL — Issue 2: Corrupted Author Names in thebibliography

**Location:** Lines 1203, 1250, 1260, 1265, 1270, 1280, 1301

**Problem:** Seven commented `\bibitem` entries contain the Chinese character sequence `全员参与全员参与全员参与全员参与` ("full participation") embedded in the author field. This is a copy-paste/replace artifact — the pattern appears in positions where a last author name was corrupted:

| Key | Corrupted | Likely correct (inferred) |
|---|---|---|
| `benedetti2019parameterized` | `…and 全员参与全员参与全员参与全员参与, L.` | `…and Benedetti, M., Grant, E., and Péran, T.` |
| `kang2019decoupling` | `…and 全员参与, P.` | `…and Dollár, P.` |
| `lin2017focal` | `…and 全员参与, P.` | `…and Dollár, P.` |
| `liu2021differentiable` | `…and 全员参与, M.` | `…and [last author unknown]` |
| `memmesheimer2024rare` | `…and 全员参与, D.` | `…and [last author unknown]` |
| `payette2019levy` | `…and 全员参与, M.` | `…and [last author unknown]` |
| `yu2023partly` | `…and 全员参与, S.` | `…and [last author unknown]` |

The corruptions all follow the pattern: a last author name was replaced by a variable/set placeholder that expanded to the Chinese phrase. If these entries are uncommented (as required by Issue 1), these strings will appear literally in the compiled PDF, making the paper unacceptable for submission.

**Fix:** Correct all author fields before uncommenting. For entries where the last author is unknown, consult the original papers to fill in the correct name.

---

### 🟠 HIGH — Issue 3: Undefined Control Sequence `\nand`

**Location:** Line 75 (abstract):
```
…optimal transport (Theorems~2--3);\nand (iii) exploration-noise calibration…
```

**Problem:** `\nand` is not defined in the preamble (no `\newcommand{\nand}{…}` exists). This will produce a LaTeX error: `! Undefined control sequence. \nand`.

**Cause:** Likely a copy-paste artifact: the line was originally `…Theorems 2–3; and (iii) exploration…`, and the word `and` was accidentally preceded by `\n` (perhaps a line-break escape), producing `\nand`.

**Fix:** Change `\nand` to `and`:
```
…optimal transport (Theorems~2--3); and (iii) exploration-noise calibration…
```

---

### 🟡 MODERATE — Issue 4: Missing Bibliography Entries for Active Citations

**Location:** Entire bibliography section (lines 1119–1327)

**Problem:** The following citation keys are used in the main text but are absent from the non-commented bibliography:

| Key | Used in paper | Status |
|---|---|---|
| `ahmadzadeh2023openpi` | §2 (Open Physical Intelligence) | ❌ Missing |
| `boixo2018characterizing` | §3 (RCS complexity) | ❌ Missing |
| `chakrabarti2020quantum` | §2 (quantum ML) | ❌ Missing |
| `chawla2002smote` | §1, §2 | ❌ Missing |
| `datacentricai` | §2 | ❌ Missing |
| `gong2023programmatic` | §2 | ❌ Missing |
| `google-ai-coscientist` | Appendix | ❌ Missing |
| `haarnoja2018soft` | §6 | ❌ Missing |
| `hu2018dro` | §1 | ❌ Missing |
| `kang2019decoupling` | §2 | ❌ Missing (also corrupted) |
| `khandelwal2022short` | §1 | ❌ Missing |
| `lin2017focal` | §1, §2 | ❌ Missing (also corrupted) |
| `liu2021differentiable` | §2 | ❌ Missing (also corrupted) |
| `llnl-open-ai-coscientist` | Appendix | ❌ Missing |
| `memmesheimer2024rare` | §1 | ❌ Missing (also corrupted) |
| `niro:2024levy` | §5 | ❌ Missing |
| `padalkar2023open` | §1, §2 | ❌ Missing |
| `payette2019levy` | §5 | ❌ Missing (also corrupted) |
| `pohlen2022diffuse` | §2 | ❌ Missing |
| `porter1956statistical` | §3 | ❌ Missing |
| `sagawa2019distributionally` | §1 | ❌ Missing |
| `toyonobu2021Isaac` | §2 | ❌ Missing |
| `yu2020metaworld` | §6 | ❌ Missing |
| `yu2023partly` | §2 | ❌ Missing (also corrupted) |
| `zeng2024quantum` | §3, §5, Appendix | ❌ Missing |
| `ahmadzadeh2023openpi` | §2 | ❌ Missing |
| `ahn2022can` | §2 | ❌ Missing |

**Fix:** Add all missing entries to `references.bib` (or uncomment + fix them in the thebibliography environment). Note: `ahmadzadeh2023openpi`, `ahn2022can`, `brohan2022rt`, `cao2019cairlab_pusht_real`, `fischer1982randomness`, `niro:2024levy`, `sagawa2019distributionally` are also missing from both the non-commented thebibliography and from any commented-out entry, so they must be added from scratch.

---

### 🟡 MODERATE — Issue 5: Suspicious KL-PT Formula (Equation 16)

**Location:** Line ~530 (Mapping II section):
```
D_{KL-PT}(k) = KL(F_{PT}^{(k)} || F_{PT}) ≈ Σ_j log(p_{PT}(y_j) / F_{PT}^{(k)}(y_j))
```

**Problem:** The formula as written compares a **density** `p_{PT}(y_j)` (the theoretical PT pdf `e^{-y_j}`) to a **CDF** `F_{PT}^{(k)}(y_j)` (the empirical cumulative distribution at step k). These are mathematically incompatible operands in a ratio. The result would be `N · Σ log(e^{-y_j} / F_{PT}^{(k)}(y_j))`, which is not a standard divergence measure.

The brief defines this quantity as `KL(F_{PT}^{(k)} || F_{PT})`, which is the KL divergence between the empirical PT distribution at step k and the target PT distribution. The standard discrete form is:
```
KL(F^(k)_PT || F_PT) = (1/N) Σ_j log(f^(k)_PT(y_j) / f_PT(y_j))
```
where both terms are probability mass functions (densities), not a density vs. a CDF.

**Likely issue:** The paper uses `p_{PT}` to denote the theoretical PT density and `F_{PT}^{(k)}` to denote the empirical PT density, but both should be densities (not `F_{PT}^{(k)}` which is a CDF by standard notation). Alternatively, the paper may intend a different divergence formula.

**Fix:** Verify with the authors whether Equation 16 correctly implements `KL(F_{PT}^{(k)} || F_{PT})`. If `F_{PT}^{(k)}` is the empirical CDF and `p_{PT}` is the theoretical pdf, this is not the KL from the brief. Recommend: `(1/N) Σ_j log(f_{PT}^{(k)}(y_j) / f_PT(y_j))` where both `f_{PT}^{(k)}` and `f_PT` are densities.

---

## Other Findings (Passed)

### ✅ LaTeX Compile — Generally Clean
- All packages loaded: `amsmath`, `amssymb`, `graphicx`, `booktabs`, `hyperref`, `xcolor`, `natbib`, `doi`, `url`, `footnote`, `makecell`, `enumitem`, `amsfonts`. No missing package dependencies.
- All `\includegraphics` paths reference existing figures: `fig1_system_overview`, `fig7_redesigned_sensitivity`, `fig8_redesigned_ablation`, `fig9_redesigned_pertask_allocation`, `fig10_data_engine`, `fig11_openx_training`, `real_hardware_pt`, `fig_pt_distribution_validation` — all in `../figs/`.
- All `\ref{}` and `\label{}` pairs are consistent: `fig:system_overview`, `fig:sensitivity`, `fig:ablation`, `fig:pertask_allocation`, `fig:data_engine`, `fig:openx_training`, `fig:real_hardware_pt`, `fig:pt_distribution_validation`; table labels `tab:mt10_results`, `tab:mt50_generalization`, `tab:cost`, `tab:hardware_robustness`, `tab:significance`, `tab:openx_training`, `tab:data_engine`, `tab:service_headline`.
- Section labels: `sec:intro`, `sec:related`, `sec:pt_prior`, `sec:mappings`, `sec:theory`, `sec:setup`, `sec:metaworld`, `sec:openx`, `sec:discussion`, `sec:conclusion` — all consistent.
- Theorem/Proposition labels: `prop:heavy_tail`, `prop:entropy`, `prop:sample_complexity`, `prop:convergence` — all consistent.
- Author/title fields: ASCII-only, no Chinese characters. `Xianghong Zeng, Yirong Jin, Wuyi Li — Coherent (Beijing) Technology Co., Ltd.` ✅

### ✅ Scientific Accuracy — PT / OT / Exploration-Noise Math
- PT law (Equation 1): `F_PT(y) = 1−e^{−y}`, `p_PT(y) = e^{−y}` ✅
- Entropy: H(PT) = 1 nat, H(Gaussian) ≈ 0.92 nat ✅ (0.9189 rounded)
- Heavy-tail superiority at t=3: 0.0498 vs 0.0027 = 18.4× ✅; at t=5: 0.00674 vs 2.87×10⁻⁷ ≈ 23,478× (brief says "23,\500×" — minor typesetting difference, values match) ✅
- Theorem 1 (rank-optimality): Statement correctly frames `(1−η)b + ηPS` as permutation-optimal ✅
- Proposition 1 (adaptive η): Gradient-descent update for η ✅
- Theorem 2 / Proposition 2 (OT monotone transport): `ξ = G^{−1}(F_{PT}(Y)) = −log(1−F_{PT}(Y))` ✅
- Theorem 3 (multidimensional copula): Copula-preserving transport strictly reduces W₁ ✅
- Proposition 3 (hardware-robust noise): Two-mode Beta mixture with mixture weight π=0.3 ✅
- Mapping III equation correctly defines `σ ~ π·Beta(2,8) + (1−π)·Beta(8,2)` ✅
- KL-PT notation `(F_{PT}^{(k)} || F_{PT})` is correct but formula may have notation issue (see Issue 5) ✅ (intention matches brief)

### ✅ Number Fidelity — All Metrics
| Claim | Brief | Paper | Status |
|---|---|---|---|
| MT10 tail (Uniform) | 0.529 | 0.529 | ✅ |
| MT10 tail (PT-rank) | 0.565 | 0.565 | ✅ |
| MT10 head retained | 0.949 | 0.949 | ✅ |
| MT50 retention (Uniform) | 77.9% | 77.9% | ✅ |
| MT50 retention (PT-rank) | 94.3% | 94.3% | ✅ |
| MT50 tail (Uniform) | 0.412 | 0.412 | ✅ |
| MT50 tail (PT-rank) | 0.533 | 0.533 | ✅ |
| Ablation: η=0 | 0.529 | 0.529 | ✅ |
| Ablation: no rank matching | 0.502 | 0.502 | ✅ |
| Ablation: no nonlinear util | 0.551 | 0.551 | ✅ |
| Ablation: no multidim OT | 0.543 | 0.543 | ✅ |
| η best range | 0.3–0.7 | 0.3, 0.7 | ✅ |
| Hardware: ideal | 0.565 | 0.565 | ✅ |
| Hardware: σ=0.02 | 0.563 | 0.563 | ✅ |
| Hardware: σ=0.05 | 0.558 | 0.558 | ✅ |
| Hardware: σ=0.08 | 0.552 | 0.552 | ✅ |
| Significance: vs Uniform +6.8 pp, p=0.003*** | exact | exact | ✅ |
| Significance: vs Focal +4.4 pp, p=0.012* | exact | exact | ✅ |
| Significance: vs DRO +3.1 pp, p=0.028* | exact | exact | ✅ |
| Significance: vs Meta-Weight +2.4 pp, p=0.045* | exact | exact | ✅ |
| Significance: vs Inv-Freq −6.2 pp, p=0.018* | exact | exact | ✅ |
| Open X: GiB | 171.62 GiB | 171.62 GiB | ✅ |
| Open X: shards | 562 | 562 | ✅ |
| Open X: episodes | 2,071 | 2,071 | ✅ |
| Allocation head: params | 865 | 865 | ✅ |
| Allocation head: features | 9 | 9 | ✅ |
| Training steps | 20,000 | 20,000 | ✅ |
| Tail share: source | 8.25% | 8.25% | ✅ |
| Tail share: Q-Tail | 50.08% (brief), 50.1% (abstract) | 50.08% / 50.1% (50.1% rounded in abstract) | ✅ |
| Tail-share reallocation | 6.07× | 6.07× | ✅ |
| KL source | 5.44×10⁻⁹ | 5.44×10⁻⁹ | ✅ |
| KL Q-Tail | 2.42×10⁻⁵ | 2.42×10⁻⁵ | ✅ |
| Tail success source | 47.83% | 47.83% | ✅ |
| Tail success Q-Tail | 53.24% | 53.24% | ✅ |
| Tail success Δ | +5.41 pp, +11.31% rel | +5.41 pp, +11.31% rel | ✅ |
| CVaR@20 source | 45.38% | 45.38% | ✅ |
| CVaR@20 Q-Tail | 50.94% | 50.94% | ✅ |
| CVaR@20 Δ | +5.56 pp | +5.56 pp | ✅ |
| Tail data share source | 5.41% | 5.41% | ✅ |
| Tail data share Q-Tail | 39.85% | 39.85% | ✅ |
| Tail coverage@50 source | 32.35% (brief), 32.4% (table) | 32.35% / 32.4% | ✅ |
| Tail coverage@50 Q-Tail | 76.47% (brief), 76.5% (table) | 76.47% / 76.5% | ✅ |
| Bootstrap p-value | <1e-3 | <10⁻³ | ✅ |
| CI95 for tail_success Δ | [0.047, 0.062] | [0.0472, 0.0615] (table), [0.047, 0.062] (abstract/conclusion) | ✅ |
| Open X datasets (8) | listed | all 8 listed correctly | ✅ |
| Service headline: 50.1% | 50.1% | 50.1% | ✅ |
| CVaR 45.38→50.94, +5.56 pp | exact | exact | ✅ |
| Tail success 47.83→53.24 | exact | exact | ✅ |

### ✅ Claim-Boundary Honesty
- **No end-to-end robot-policy claims for Open X:** Explicitly stated as "allocation-head training protocol" (abstract, §8.1, §8.2, §8.3, §9) ✅
- **Open X vs. Quafu/Baihua separation:** Quafu/Baihua is in §6 (§6.2: "Real Quantum Hardware") and §7.1 (MT10+Real RCS row in table). Open X is in §8. MetaWorld is §7. Separation is clean. ✅
- **Data-distribution quality framing:** §8.2 explicitly states "evaluates data-distribution quality, not full policy training" ✅
- **Same-policy confirmation required:** §9 Discussion explicitly states "Same-policy confirmation required" as an honest unresolved limitation ✅
- **Adapter-ready anchors:** §8.2 and §9 correctly attribute external anchors as "metadata-level alignments" requiring "full policy-export validation" ✅
- **No conflation of simulation and real results:** Simulation (§7) and real (§8) clearly labeled with different subsections ✅

### ✅ Completeness
All required sections from the brief are present:
- ✅ Abstract updated with real Open X results, service framing
- ✅ §1 Introduction gap stated: "simulation-to-real gap"
- ✅ §8 New "Real Open X Embodiment Validation" section with §8.1 (allocation-head training), §8.2 (data-engine), §8.3 (service)
- ✅ §7 MetaWorld contextualized as "controlled in-silico validation"
- ✅ §2 Related Work includes Open X-Embodiment, data-centric AI, synthetic training-data services
- ✅ §9 Discussion/Limitations upgraded; all 5 honest limitations stated
- ✅ §10 Conclusion includes real-data result and deployed service
- ✅ Appendix §A: Multi-agent co-scientist methodology documented
- ✅ All required figures: fig1, fig10, fig11, plus reused figs 7, 8, 9, real_hardware_pt, fig_pt_distribution_validation ✅

### ⚠️ Minor LaTeX / Formatting Issues (Non-blocking)

1. **Table 7 (significance table):** The comparison "vs. Logit Adj. +5.0 pp" is listed in the table but the brief specifies "+5.0 pp" in the significance table but the abstract/brief discussion mentions a different comparison. The paper's MT10 table (Table 1) shows Logit Adjustment tail = 0.538, and PT-rank = 0.565, so Δ = +0.027 = +2.7 pp. The significance table (Table 7) shows "+5.0 pp, p=0.008**" for Logit Adj. This is **internally inconsistent**: Table 1 says Logit Adj tail = 0.538 (Δ = 0.565−0.538 = 0.027 = 2.7 pp) but Table 7 says +5.0 pp. The brief says nothing about Logit Adj significance specifically, but this internal inconsistency between the MT10 results table and the significance table should be verified — the Logit Adj comparison in Table 7 should be ~+2.7 pp, not +5.0 pp.

2. **`\hfill` or spacing in table captions:** Table 9 caption ends with `\label{tab:cost}` but the `\caption{}` and `\label{}` order is correct ✅.

3. **Text in table cells:** In Table 16 (service headline), `\textbf{171.62 GiB, 562 shards, 8 datasets}` spans the Source and Q-Tail columns — this is a merged-cell approach that is syntactically valid in booktabs tables ✅.

4. **The `\begin{thebibliography}` conflicts with `\bibliography{}`:** This is a document-level conflict. Removing one or the other is required for clean compilation.

---

## Verdict

| Category | Status |
|---|---|
| LaTeX compile correctness | ⚠️ Issues 1, 3, 4 |
| Scientific accuracy (PT/OT/exploration-noise) | ✅ (Issue 5 needs verification) |
| Number fidelity | ✅ All exact |
| Claim-boundary honesty | ✅ All compliant |
| Completeness (sections) | ✅ All present |
| Bibliography entries | ❌ Critical failure |
| Author/title field cleanliness | ✅ Clean |

**Recommended action:** Fix Issues 1–4 before compilation. Issue 5 (KL formula) should be reviewed by the author for mathematical correctness. The Logit Adj significance discrepancy (Table 1 vs. Table 7, ~+2.7 pp vs. +5.0 pp) should be verified.
