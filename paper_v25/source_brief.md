# Source Brief — Q-TAIL Paper Revision (ver.24 → ver.25)

**Goal:** Rewrite "Quantum Random Circuit Sampling as a Distributional Prior for Long_Tail Embodied Learning"
using (a) the existing ver.24 manuscript and (b) the LATEST project data from the running Q-TAIL-MVP
service (quantum-embodied-data-service, qtail-openx-training, qtail-data-engine). Produce a single
coherent manuscript in PDF + Word. The revision is driven by a multi-agent AI co-scientist pipeline
(Generation → Reflection → Ranking → Evolution → Proximity → Meta-review + Supervisor), adapted from
the open-source framework at github.com/llnl/open-ai-co-scientist (Google AI co-scientist architecture).

**Authors (unchanged):** Xianghong Zeng, Yirong Jin, Wuyi Li — Coherent (Beijing) Technology Co., Ltd.

---

## A. What ver.24 already contains (KEEP, lightly edit)

### Core idea
Random-circuit sampling (RCS) yields output probabilities whose chaotic-limit statistics follow the
Porter-Thomas (PT) law: Y = N·p_U(x) → Exp(1), F_PT(y)=1−e^{−y}. PT samples are a *distributional prior*
(transformed, not semantic). KEY CLARIFICATION (ver.24 already states): PT can be classically simulated
with exponential variates; the contribution is NOT quantum speedup but a theoretically grounded,
analytically tractable heavy-tail prior.

### Three training-time distributions (Theorem/Proposition backbone — KEEP)
- **Mapping I — Sample scheduling:** q = (1−η)·b + η·P·S, rank-optimal assignment (Theorem 1,
  permutation-optimal rank matching). η∈[0,1] mixing coefficient; adaptive update (Proposition 1).
- **Mapping II — Risk-scene:** ξ = G⁻¹(F_PT(Y)); monotone transport minimizes 1-Wasserstein
  (Theorem 2 / Proposition 2). Multidim via Copula (Theorem 3).
- **Mapping III — Exploration noise:** σ = H⁻¹(F_PT(Y)), H a two-mode Beta mixture (Proposition 3
  hardware robustness bound).

### Theory (KEEP)
- Proposition 4 (heavy-tail superiority): P(Y_PT>t)=e^{−t} > P(|Z|>t) for t≥1.5; at t=3, 0.0498 vs 0.0027.
- Proposition 5 (entropy): H(PT)=1 nat > H(Gaussian baseline)≈0.92 nat.
- Theorem 4 (sample-complexity lower bound) and Proposition 6 (policy convergence under contraction).
- Connection to power-law / Lévy flights (Section 7.3).

### Meta-World simulation results (KEEP as controlled simulation)
- 5 seeds {42,123,456,789,1024}; SAC; 100k steps/seed; 100 eval episodes / 5k steps.
- MT10: Uniform tail 0.529 → PT-rank 0.565; head retained 0.949; overall 0.818; CVaR@20 0.548.
  Real RCS (Quafu/Baihua, 15 qubits, depth 28, ~196 CNOT, 100k shots): tail 0.552 (3.2% degradation).
- MT50 generalization: Uniform retention 77.9% → PT-rank 94.3%; MT50 tail 0.412 → 0.533.
- Baselines (MT10 tail): Focal Loss 0.541, Logit Adj 0.538, DRO 0.548, Meta-Weight 0.552,
  Inv-Freq 0.602 (but head drops to 0.930), Empirical 0.176.
- η sensitivity: best balance 0.3 ≤ η ≤ 0.7 (Table 9 / Fig 8).
- Ablation (Table 8): Full 0.565; remove PT prior (η=0) 0.529; remove rank matching 0.502;
  remove nonlinear util 0.551; remove multidim OT 0.543.
- Relative cost: PT-rank 1.3× vs DRO 2.5×, Meta-Weight 3.8×.
- Hardware robustness (Table 10): ideal 0.565, real RCS 0.552, noise σ=.02 0.563, σ=.05 0.558, σ=.08 0.552.
- Significance (Table 7): PT-rank vs Uniform +6.8% p=0.003***; vs Focal +4.4% p=0.012*;
  vs DRO +3.1% p=0.028*; vs Meta-Weight +2.4% p=0.045*; vs Inv-Freq −6.2% p=0.018*.

### Stated limitations in ver.24 (these are now PARTIALLY addressed by new data)
"simulation-only Meta-World validation", "simplified hardware-noise modeling", "incomplete tests on
navigation, locomotion, and high-dimensional real-robot systems."

---

## B. NEW project data to incorporate (from localhost:6222 service + JSON reports)

### B1. quantum-embodied-data-service (Q-Tail Synthetic Training Data Service) — PRODUCTIZED
- Private Preview 2026, running **Strong Open X snapshot**.
- Real Open X data ingested: **171.62 GiB, 8 embodied datasets, full TFRecord, 562 shards parsed 100%,
  2,071 real episodes record-level feature extraction, final training 20,000 steps (Strong checkpoint).**
- Service flow: (1) ingest customer data → (2) long-tail & risk profiling → (3) PT heavy-tail calibration
  using Open X-trained quantile gain curve → (4) deliver synthetic data plan + audit package.
- Headline same-budget evidence (response model `response_success_v1`, budget 100,000):
  - **Tail data allocation: 8.25% → 50.08% (+41.83 pp, 6.07×).**
  - **Tail success (protocol eval): 47.83% → 53.24% (+5.41 pp, relative +11.31%).**
  - **CVaR@20: 45.38% → 50.94% (+5.56 pp).**
  - **Tail data share coverage: 5.41% → 39.85% (+34.43 pp).**
  - MetaWorld anchor: +10.46 pp tail success (50 tasks).
- API: `GET /health`, `POST /generate` (customer CSV → PT plan + delivery package),
  `GET /api-docs`, `POST /access-requests`. Local preview at 127.0.0.1:8223; production creds post-approval.
- Delivery artifacts: task_profiles.csv, qtail_service_synthetic_plan.csv, per_task_comparison.csv,
  model card, package manifest (SHA256), qtail_delivery_package.zip, README.

### B2. qtail-openx-training (#objective) — REAL OPEN X ALLOCATION-HEAD TRAINING
Source: results/openx_strong_training/openx_demo_training_report.json (generated 2026-07-10).
- **8 datasets:** austin_buds, austin_sirius, berkeley_mvp, columbia_cairlab_pusht_real, language_table,
  language_table_sim, nyu_door_opening, ucsd_kitchen (all converted to RLDS).
- 562 TFRecord shards, **184.3 GB (171.622 GiB)**, 20,000 training steps.
- Allocation head: **865 parameters**, 9 features (log_bytes, shard_size_rarity, dataset_frequency,
  shard_position, mean_episode_steps, reward_failure_proxy, action_complexity, instruction_complexity,
  terminal_rate). Identical architecture/optimizer/steps/seed for source & Q-Tail heads.
- Trajectory evidence: 562/562 shards parsed (record_parse_rate 1.0), 2,071 records decoded
  (cap 4/shard), mean episode steps 127.20.
- Tail definition: top 30% by record-informed tail score.
  - **source_tail_share 0.0825 → qtail_tail_share 0.5013 (+41.88 pp, ≈6.07×).**
  - predicted: source 0.0825 → qtail 0.5008 (consistent with PT tail goal = true).
- KL to target tail law converges: source 5.44e-9, Q-Tail 2.42e-5 at step 20k (both → ~0).
- **Claim boundary (verbatim, MUST be honored):** real Open X *record-informed allocation-head*
  training; every complete shard covered; it is NOT full robot-policy training and does NOT prove
  downstream policy success without a same-policy run. Both heads identical except the PT prior.

### B3. qtail-data-engine — USER DATA vs PT SYNTHETIC (same-budget protocol)
Source: results/qtail_openx_service_public/qtail_data_engine_report.json.
- 114 task profiles, 34 tail tasks (top 30% by tail score). Same total budget 100,000.
  Same response model `response_success_v1`. Same metric set
  {overall_success, tail_success, cvar20, extreme_failure_count, tail_coverage_at_50, tail_data_share}.
  Allocation sums both = 1.0. PT source = 29,581 rows, Gini 0.339.
- Metrics (source → Q-Tail synthetic):
  - overall_success 0.6548 → 0.6725 (+1.78 pp)
  - **tail_success 0.4783 → 0.5324 (+5.41 pp, relative +11.31%)**
  - **cvar20 0.4538 → 0.5094 (+5.56 pp)**
  - **tail_data_share 0.0541 → 0.3985 (+34.43 pp)**
  - **tail_coverage_at_50 0.3235 → 0.7647 (+44.12 pp)**
- Significance (paired task-level bootstrap, 5,000 iters):
  - tail_success delta 0.0541, CI95 [0.0472, 0.0615], p(Δ≤0)=0.0, positive_pair_rate 1.0
  - overall_success delta 0.0178, CI95 [0.0116, 0.0238], p=0.0
  - cvar20 delta 0.0556, CI95 [0.0476, 0.0630], p=0.0
- Decision gate: winner = qtail_synthetic, test_passed = true (gains exceed min gates; p within bound).
- External validation anchors (adapter-ready): Open X-Embodiment/RT-X, Meta Habitat 3.0,
  DROID/BridgeData.
- **Claim boundary (MUST be honored):** evaluates *data-distribution quality*, not full policy training;
  synthetic rows are allocation targets / scenario specs; rendering or robot execution is a downstream
  adapter. Public anchors are aggregate-metadata validations unless full exports supplied.

---

## C. Required changes to the manuscript

1. **Abstract:** add the real Open X validation result (171.62 GiB / 562 shards / 2,071 episodes;
   tail allocation 6.07×; tail success +5.41 pp, rel +11.31%; CVaR +5.56 pp; all p<0.001 paired bootstrap)
   and the production data-service framing. Keep MetaWorld numbers.
2. **Intro (§1):** state the gap (simulation-only) and how real Open X evidence + data service closes it.
3. **New §:** "Real Open X Embodiment Validation" — allocation-head training (B2) + data engine
   same-budget eval (B3) + production data service (B1). Use new Tables 14–16 and Figs 1, 10, 11.
4. **Keep MetaWorld (§8) as controlled simulation**, now contextualized as the in-silico controlled study.
5. **Related Work:** add Open X-Embodiment / large-scale robot datasets; data-centric ML; synthetic
   training-data services.
6. **Discussion / Limitations:** upgrade — simulation-only limitation now addressed by real Open X;
   state remaining honest boundaries (data-distribution quality not end-to-end policy training;
   same-policy confirmation still required; anchors adapter-ready).
7. **Conclusion:** include the real-data result and the deployed service as the practical payoff.
8. **Methodology appendix (new, optional):** describe the multi-agent co-scientist revision process
   (Generation/Reflection/Ranking/Evolution/Proximity/Meta-review + Supervisor) — novel and on-theme.

## D. HONESTY / CLAIM-BOUNDARY RULES (non-negotiable)
- Do NOT claim end-to-end robot-policy success. The Open X results are (i) allocation-head training
  convergence and (ii) same-budget *data-distribution* quality gains from a response-model protocol.
- Keep exact numbers from §A and §B; do not invent metrics. Round consistently (e.g., 50.1% for tail share).
- Cite Quafu/Baihua real-hardware result (ver.24) AND new Open X result separately; do not conflate.
- Attribute the multi-agent revision process honestly in the methodology note.

## E. Figures available (paper_v25/figs/)
- fig1_system_overview.png  (NEW — full pipeline: PT source → 3 distributions → MetaWorld sim + Open X real → allocation head → data engine/API)
- fig7_redesigned_sensitivity.png, fig8_redesigned_ablation.png, fig9_redesigned_pertask_allocation.png (ver.24, reuse)
- real_hardware_pt.png (Quafu/Baihua ideal vs real, reuse)
- fig_pt_distribution_validation.png (PT distribution validation, reuse)
- fig10_data_engine.png (NEW — source vs Q-Tail synthetic bars)
- fig11_openx_training.png (NEW — tail-share reallocation + KL convergence)
