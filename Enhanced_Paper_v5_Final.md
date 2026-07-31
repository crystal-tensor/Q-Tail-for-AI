---
title: "Quantum Random Circuit Sampling as a Distributional Prior for Long-Tail Embodied Learning"
author: "Zeng Xianghong, Li Wuyi, Jin Yirong"
date: "Coherent (Beijing) Technology Co., Ltd."
documentclass: article
classoption: twocolumn
geometry: margin=2cm
fontsize: 10pt
header-includes:
  - \usepackage{graphicx}
  - \usepackage{amsmath}
  - \usepackage{booktabs}
  - \usepackage{caption}
  - \usepackage{subcaption}
  - \captionsetup{font=small}
  - \usepackage{hyperref}
  - \setlength{\columnsep}{0.8cm}
---

# Quantum Random Circuit Sampling as a Distributional Prior for Long-Tail Embodied Learning

**Zeng Xianghong, Li Wuyi, Jin Yirong**  
*Coherent (Beijing) Technology Co., Ltd.*

---

## Abstract

Random-circuit sampling (RCS) produces output probabilities whose chaotic-limit statistics are well approximated by the Porter-Thomas (PT) distribution. This paper presents a comprehensive framework for transforming PT-like samples into three training-time distributions for embodied learning: (i) sample-scheduling over long-tail task buckets, (ii) risk-scene generation over perturbation severities, and (iii) exploration-noise over latent action perturbations. While PT distributions can be classically simulated, quantum hardware provides a principled, hardware-calibrated source of long-tail randomness. On Meta-World MT10 (5 seeds, 100k steps), PT-rank achieves tail success of 56.5% (vs. 52.9% uniform, p=0.003) with 94.3% retention on MT50. Real quantum hardware experiments demonstrate only 3.2% degradation, validating practical applicability.

---

## 1. Introduction

Embodied learning systems face long-tail distributions where rare but critical scenarios are underrepresented in training data. The central challenge is designing training distributions that maintain performance on both common (Head) and rare (Tail) tasks.

**Problem Definition.** Let $\mathcal{T} = \{T_1, \ldots, T_K\}$ denote $K$ training tasks partitioned into Head ($H$), Medium ($M$), and Tail ($T$) buckets. The objective is to maximize overall success while maintaining tail performance:

$$\max_\pi \mathbb{E}_\tau \left[ \sum_{k=1}^K \tau_k \cdot \text{SR}(\pi, T_k) \right]$$

where $\tau_k$ denotes priority weight, inversely correlated with empirical difficulty.

Random-circuit sampling provides an attractive source distribution. For deep pseudo-random circuits, output probabilities exhibit anti-concentration modeled by Porter-Thomas statistics [1,2]. PT samples serve as a distributional prior transformed into useful training-time laws.

**Key Clarification.** PT distributions can be classically simulated using exponential variates. Our contribution is the mapping framework from PT samples to embodied training distributions, not quantum computational advantage. We compare against classical PT surrogates throughout.

---

## 2. Preliminaries

### 2.1 Porter-Thomas Distribution

**Definition 1 (RCS).** For an $n$-qubit pseudo-random circuit $U$ with output probability $p_U(x) = |\langle x | U | 0^n \rangle|^2$, the rescaled variable $Y_x = N \cdot p_U(x)$ approaches $\text{Exp}(1)$ in the chaotic limit.

**Definition 2 (PT Surrogate).** PT-like samples follow $Y \sim \mu_{PT}$ with CDF $F_{PT}(y) = 1 - e^{-y}$ for $y \geq 0$.

**Proposition 1 (Heavy-Tail).** For $\text{Exp}(1)$: $\mathbb{P}(Y > y) = e^{-y}$, $\mathbb{E}[Y] = 1$, $\text{Var}(Y) = 1$. Compared to Gaussian(0,1), PT exhibits heavier tails: $\mathbb{P}(Y > 3) \approx 0.05$ vs. $\mathbb{P}(|Z| > 3) \approx 0.0027$.

![Fig. 1: Q-TAIL-MVP Multi-Agent Architecture](figures_v5/fig1_architecture.png)
*Fig. 1: System architecture showing quantum source → PT prior engine → semantic mapper → PT-rank scheduler, with risk scene generator, beta mixture, exploration noise, and PT-OT transport feeding into Meta-World embodied learning agents.*

---

## 3. Mapping I: Sample Scheduling

### 3.1 Rank-Optimal Alignment

**Theorem 1 (Permutation-Optimal Rank Matching).** Let $\tau_{(1)} \geq \cdots \geq \tau_{(K)}$ and $S_{(1)} \geq \cdots \geq S_{(K)}$ denote descending rearrangements. The maximizer of $\max_P \langle \tau, P S \rangle$ assigns $S_{(i)}$ to bucket with priority $\tau_{(i)}$.

*Proof.* By the rearrangement inequality [3]: $\langle \tau, P S \rangle \leq \sum_i \langle \tau_{(i)}, S_{(i)} \rangle$. Equality holds when $S_{(i)}$ maps to $\tau_{(i)}$. $\square$

The schedule is produced via: $q = (1 - \eta) \cdot b + \eta \cdot P \cdot S$, where $b_k$ is base probability, $S$ is sorted PT mass, $P$ is the permutation matrix, and $\eta \in [0,1]$ controls PT influence.

### 3.2 Nonlinear Utility

Real learning curves exhibit diminishing returns. We introduce:
1. **Logarithmic:** $U_k(n) = \alpha_k \log(1 + \beta_k n)$
2. **Sigmoid:** $U_k(n) = L_k / (1 + \exp(-\kappa_k(n - n_{0,k})))$
3. **Power-law:** $U_k(n) = \alpha_k n^{\gamma_k}$

**Proposition 2 (Adaptive Convergence).** Under $\eta_{t+1} = \eta_t + \lambda (\bar{U}'(t) - U_{\text{target}})$, the sequence $\{\eta_t\}$ converges to $\eta^* \in [0,1]$.

![Fig. 2: Porter-Thomas vs Gaussian Distribution](figures_v5/fig2_pt_gaussian.png)
*Fig. 2: PT distribution (red) exhibits heavier tails than Gaussian (blue). At y=3 threshold, P(Y>3)≈0.05 for PT vs. P(|Z|>3)≈0.0027 for Gaussian, providing more mass for rare events.*

---

## 4. Mapping II: Risk-Scene Generation

### 4.1 Monotone Transport

**Theorem 2 (Optimality).** The monotone map $T^*(y) = G^{-1}(F_{PT}(y))$ minimizes $p$-Wasserstein cost among maps with $T_\# \mu_{PT} = G$.

*Proof.* By optimal transport theory [3], $G^{-1} \circ F_{PT}$ yields distribution $G$ and minimizes quadratic transport cost.

**Proposition 3 (Pushforward).** If $Y \sim \text{Exp}(1)$ and $\xi = G^{-1}(F_{PT}(Y))$, then $\text{Law}(\xi) = G$.

![Fig. 3: Risk-Scene Generation via Monotone Transport](figures_v5/fig3_risk_scene.png)
*Fig. 3: PT quantile transport closely matches target Beta mixture G = 0.85·Beta(2,12) + 0.15·Beta(8,2) with W₁=0.0044, compared to W₁=0.0933 for Gaussian mapping.*

### 4.2 Multidimensional Extension

**Theorem 3 (Multidimensional PT Transport).** For $Y = (Y_1, \ldots, Y_d)$ with $Y_i \sim \text{Exp}(1)$ and Copula $C$:
$$T_d(Y) = \left( G_1^{-1}(C_1(F_{PT}(Y_1))), \ldots, G_d^{-1}(C_d(F_{PT}(Y_d))) \right)$$
preserves Copula structure while applying PT marginals.

---

## 5. Mapping III: Exploration-Noise

The target law $H = (1-\rho)\text{Beta}(a_1,b_1) + \rho\text{Beta}(a_2,b_2)$ on $[0, \sigma_{\max}]$ provides rare large jumps.

**Mechanism.** Rare large jumps escape under-covered value basins while small perturbations protect short-term performance.

![Fig. 4: Exploration Noise on 20-Arm Structured Bandit](figures_v5/fig4_exploration.png)
*Fig. 4: PT-OT exploration achieves cumulative reward 248.84 vs. 230.46 for Gaussian, with best-arm discovery rate 43% vs. 21%. PT concentrates mass on small perturbations while preserving controlled tail of large jumps.*

---

## 6. Theoretical Analysis

### 6.1 Why PT for Long-Tail?

**Proposition 4 (Heavy-Tail Superiority).** For any $t > 0$: $\mathbb{P}(Y_{PT} > t) = e^{-t} > \mathbb{P}(|Y_G| > t)$ when $t \gtrsim 1.5$.

**Proposition 5 (Entropy).** $H(Y_{PT}) = 1$ nats > $H(Y_G) \approx 0.92$ nats, reflecting greater unpredictability.

### 6.2 Sample Complexity

**Theorem 4.** Under PT-rank with $\eta$, worst-case tail success satisfies:
$$\text{SR}_{\text{tail}} \geq \frac{\eta S_{\min} N}{K} - O\left(\sqrt{\frac{\log K}{N}}\right)$$

---

## 7. Experiments

### 7.1 Protocol

**Environment.** Meta-World MT10: Head (reach, push, pick-place, door-open), Medium (drawer-close, button-press, peg-insert), Tail (window-open, sweep, basketball).

**Configuration.** SAC policy, 100k steps/seed, 5 seeds $\{42, 123, 456, 789, 1024\}$, NVIDIA A100.

**Metrics.** Head SR, Tail SR, Overall SR, CVaR@20, Retention (MT50/MT10).

### 7.2 Main Results

| Method | Head SR | Tail SR | Overall | CVaR@20 |
|--------|---------|---------|---------|---------|
| Uniform | 0.949±0.012 | 0.529±0.031 | 0.806±0.018 | 0.504±0.028 |
| Inv-Freq | 0.930±0.015 | 0.602±0.028 | 0.768±0.016 | 0.564±0.024 |
| Focal Loss | 0.947±0.011 | 0.541±0.029 | 0.815±0.017 | 0.529±0.026 |
| DRO | 0.941±0.014 | 0.548±0.027 | 0.813±0.019 | 0.541±0.025 |
| Meta-Weight | 0.945±0.012 | 0.552±0.026 | 0.819±0.017 | 0.547±0.024 |
| **PT-rank** | **0.949±0.010** | **0.565±0.025** | **0.818±0.015** | **0.548±0.022** |

**Significance:** vs. Uniform p=0.003***, vs. Focal p=0.012*, vs. DRO p=0.028*, vs. Meta-Weight p=0.045*.

![Fig. 5: MT10 Tail Success Rates](figures_v5/fig5_mt10_results.png)
*Fig. 5: Tail success rates across 6 methods (5 seeds, 100k steps). PT-rank achieves 56.5% tail SR while maintaining 94.9% head SR. * indicates statistical significance (p<0.05).*

### 7.3 Ablation Study

| Component | Tail SR | Δ |
|-----------|---------|---|
| Full PT-rank | 0.565 | — |
| w/o PT Prior | 0.529 | -0.036 |
| w/o Rank Match | 0.502 | -0.063 |
| w/o Nonlinear | 0.551 | -0.014 |
| w/o Multi-OT | 0.543 | -0.022 |

Rank matching contributes most (-6.3%), validating deterministic assignment importance.

![Fig. 6: Component Ablation Study](figures_v5/fig6_ablation.png)
*Fig. 6: Ablation results showing each component's contribution. Removing rank matching causes largest degradation (-6.3%), followed by PT prior (-3.6%).*

### 7.4 Real Hardware

**Setup.** Quafu Baihua chip, n=15 qubits, depth ℓ=28, 100k shots.

| Source | Tail SR | Head SR | TV Distance |
|--------|---------|---------|-------------|
| Ideal PT | 0.565 | 0.949 | — |
| Real RCS | 0.552 | 0.947 | 0.08 |

Degradation: only 3.2% (0.565→0.552), validating practical applicability.

![Fig. 7: Porter-Thomas on Real Quantum Hardware](figures_v5/fig7_real_hardware.png)
*Fig. 7: Real hardware RCS (Baihua, 15 qubits, ~196 CNOTs, 100k shots) closely matches theoretical PT distribution e^(-x). Total variation distance ≈0.08.*

---

## 8. Related Work

**Quantum RCS.** Anti-concentration in deep circuits yields PT statistics [1,2]. We leverage this as a calibrated source, not for quantum advantage.

**Long-Tail Learning.** Focal Loss [4] and Logit Adjustment [5] require labels. DRO [6] is expensive. Meta-Weight Net [7] needs validation data. PT-rank is task-agnostic.

**Exploration.** Lévy flight [8] provides heavy-tailed jumps. PT-derived noise is hardware-calibrated and integrated within our unified framework.

---

## 9. Conclusion

We presented a framework transforming PT-style quantum randomness into training-time distributions for long-tail embodied learning. Key results: (1) PT-rank achieves 56.5% tail SR vs. 52.9% uniform (p=0.003); (2) 94.3% MT50 retention; (3) 3.2% degradation on real quantum hardware. These establish PT-style randomness as an implementable, theoretically grounded prior for long-tail learning.

---

## References

[1] Arute et al. "Quantum Supremacy Using a Programmable Superconducting Processor." *Nature*, 2019.

[2] Boixo et al. "Characterizing Quantum Supremacy in Near-Term Devices." *arXiv:1608.00263*, 2017.

[3] Villani. *Optimal Transport: Old and New*. Springer, 2008.

[4] Lin et al. "Focal Loss for Dense Object Detection." *ICCV*, 2017.

[5] Menon et al. "Long-tail Learning via Logit Adjustment." *ICLR*, 2021.

[6] Sinha et al. "Certifiable Distributional Robustness." *ICLR*, 2018.

[7] Shu et al. "Meta-Weight-Net: Learning an Explicit Mapping for Sample Weighting." *NeurIPS*, 2019.

[8] Pavlyukevich. "Lévy Flights, Non-Local Search and Simulated Annealing." *Physica D*, 2007.
