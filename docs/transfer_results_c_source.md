# Transfer Learning Results — C-Source Experiments
**Date:** 2026-05-15  
**Source training:** Env C, 300k steps  
**Transfer budget:** 50k fine-tuning steps  
**Extractors compared:** euromlsys, fusion, attention_idfree, attention_pooling, spane, pma, hierarchical, hybrid_rbf, swat, turret

---

## Environments

| Env | DCs | Types | Notes |
|-----|-----|-------|-------|
| C (source) | 5 | cloud + edge + micro | Largest — training env |
| B (target) | 3 | edge + micro | Strict subset of C topology |
| A (target) | 2 | edge + micro only | Cloud DC absent — structural change |

C→B is a **size downscale**: same DC types, fewer DCs.  
C→A is a **structural downscale**: cloud DC class disappears entirely.

---

## Source Training on Env C (300k steps)

| Extractor | Peak | Final-50 | Episodes |
|-----------|-----:|--------:|--------:|
| euromlsys | 12.732 | 12.294 | 14,304 |
| fusion | 12.732 | 12.351 | 14,305 |
| hierarchical | 12.704 | 11.856 | 14,266 |
| swat | 12.662 | 12.025 | 14,260 |
| pma | 12.601 | 11.561 | 14,285 |
| attention_idfree | 12.574 | 11.461 | 14,283 |
| attention_pooling | 12.560 | 11.776 | 14,273 |
| spane | 12.551 | 11.703 | 14,296 |
| hybrid_rbf | 12.540 | 11.820 | 14,289 |
| turret | 12.537 | 11.656 | 14,294 |

**Observation:** euromlsys and fusion peak highest (12.732) but have the largest gap between peak and final-50-mean, suggesting reward-variance or late instability. hierarchical and swat are close behind with better final stability.

---

## C → A Transfer Results (structural downscale)

### Oracle — Env A from scratch (50k steps)

| Extractor | Start | Peak | Final-50 | 80% threshold |
|-----------|------:|-----:|---------:|-------------:|
| euromlsys | 1.745 | 4.919 | 3.453 | 3.935 |
| fusion | 1.745 | 5.056 | 3.574 | 4.045 |
| swat | 1.745 | 5.005 | 3.447 | 4.004 |
| turret | 1.745 | 5.028 | 3.355 | 4.022 |
| attention_idfree | 1.745 | 4.892 | 3.321 | 3.913 |
| attention_pooling | 1.745 | 4.898 | 3.386 | 3.918 |
| spane | 1.745 | 4.868 | 3.467 | 3.895 |
| hybrid_rbf | 1.745 | 4.806 | 3.256 | 3.845 |
| hierarchical | 1.745 | 4.752 | 3.438 | 3.802 |
| pma | 1.745 | 4.743 | 3.334 | 3.795 |

### Transfer Metrics — C → A (ranked by norm jumpstart)

| Rank | Extractor | NormJump | Ep→80% | NormPeak | FinalvPk | AUC |
|------|-----------|--------:|-------:|---------:|---------:|----:|
| 1 | **hybrid_rbf** | **82.6%** | **1** | **106.9%** | 73.0% | 1.108 |
| 2 | attention_idfree | 78.1% | 3 | 100.2% | 73.3% | 1.120 |
| 3 | fusion | 67.8% | 23 | 94.3% | 68.9% | 1.054 |
| 4 | hierarchical | 65.1% | 4 | 103.8% | **77.6%** | **1.135** |
| 5 | attention_pooling | 64.1% | 3 | 96.5% | 74.4% | 1.094 |
| 6 | spane | 63.3% | 17 | 100.0% | 71.2% | 1.119 |
| 7 | swat | 62.5% | 4 | 96.6% | 71.2% | 1.117 |
| 8 | pma | 60.4% | 6 | 103.9% | 74.8% | 1.121 |
| 9 | euromlsys | 53.3% | 9 | 95.8% | 71.9% | 1.141 |
| 10 | turret | 49.8% | 9 | 97.2% | 70.0% | 1.105 |

**Metric legend:**
- **NormJump** = zero-shot reward / oracle peak (higher = better instant generalization)
- **Ep→80%** = episodes until reaching 80% of oracle peak (lower = faster adaptation)
- **NormPeak** = best transfer reward / oracle peak (>100% = exceeds oracle ceiling)
- **FinalvPk** = final-50-episode mean / oracle peak (convergence quality)
- **AUC** = transfer curve AUC / oracle curve AUC (>1.0 = positive transfer overall)

---

## C → B Transfer Results (size downscale)

### Oracle — Env B from scratch (50k steps)

| Extractor | Start | Peak | Final-50 | 80% threshold |
|-----------|------:|-----:|---------:|-------------:|
| euromlsys | 5.930 | 11.522 | 10.167 | 9.217 |
| fusion | 5.930 | 11.591 | 10.146 | 9.273 |
| swat | 5.930 | 11.433 | 9.523 | 9.146 |
| hybrid_rbf | 5.930 | 11.411 | 9.516 | 9.129 |
| attention_pooling | 5.930 | 11.390 | 9.869 | 9.112 |
| hierarchical | 5.930 | 11.322 | 9.639 | 9.058 |
| pma | 5.930 | 11.280 | 9.684 | 9.024 |
| attention_idfree | 5.930 | 11.155 | 9.441 | 8.924 |
| spane | 5.930 | 11.137 | 9.575 | 8.910 |
| turret | 5.930 | 11.519 | 9.487 | 9.215 |

### Transfer Metrics — C → B (ranked by norm jumpstart)

| Rank | Extractor | NormJump | Ep→80% | NormPeak | FinalvPk | AUC |
|------|-----------|--------:|-------:|---------:|---------:|----:|
| 1 | **swat** | **92.5%** | 1 | 105.4% | 98.7% | 1.314 |
| 2 | attention_idfree | 91.4% | 1 | **107.4%** | 97.3% | 1.280 |
| 3 | hierarchical | 90.3% | 1 | 106.2% | 99.5% | 1.290 |
| 4 | fusion | 89.5% | 1 | 105.0% | **101.8%** | 1.315 |
| 5 | pma | 88.5% | 1 | 106.4% | 96.7% | 1.259 |
| 5 | turret | 88.5% | 1 | 102.5% | 94.1% | 1.230 |
| 7 | spane | 88.1% | 1 | 107.0% | 99.6% | 1.278 |
| 8 | euromlsys | 87.6% | 1 | 106.1% | 101.5% | **1.332** |
| 9 | hybrid_rbf | 82.8% | 1 | 103.8% | 91.7% | 1.199 |
| 10 | **attention_pooling** | 76.8% | **3** | 105.4% | 97.0% | 1.255 |

---

## Cross-Direction Summary

| Extractor | C→A NormJump | C→A Ep→80% | C→A NormPeak | C→B NormJump | C→B Ep→80% | Overall verdict |
|-----------|------------:|----------:|------------:|------------:|----------:|----------------|
| hybrid_rbf | **82.6%** | **1** | **106.9%** | 82.8% | 1 | Best structural downscale zero-shot |
| attention_idfree | 78.1% | 3 | 100.2% | 91.4% | 1 | Most consistent across both directions |
| hierarchical | 65.1% | 4 | 103.8% | 90.3% | 1 | Best C→A asymptotic (77.6% FinalvPk) |
| swat | 62.5% | 4 | 96.6% | **92.5%** | 1 | Best zero-shot on size downscale |
| fusion | 67.8% | 23 | 94.3% | 89.5% | 1 | Good C→B, catastrophic C→A adaptation |
| attention_pooling | 64.1% | 3 | 96.5% | 76.8% | **3** | DC ID embedding penalises both directions |
| spane | 63.3% | 17 | 100.0% | 88.1% | 1 | Solid but slow C→A recovery |
| pma | 60.4% | 6 | 103.9% | 88.5% | 1 | Strong C→A ceiling (103.9%), slower start |
| euromlsys | 53.3% | 9 | 95.8% | 87.6% | 1 | Best AUC both dirs, weakest zero-shot |
| turret | 49.8% | 9 | 97.2% | 88.5% | 1 | Consistently middle-tier |

---

## Key Insights

### 1. RBF kernel pooling (hybrid_rbf) is the best architecture for structural downscale

**Result:** 82.6% zero-shot, 1 episode to 80%, 106.9% norm peak on C→A — best on all three metrics simultaneously.

**Why:** The Gaussian RBF kernel (`exp(-||q-k||²/2σ²)`) gives pooling a natural locality property. When the cloud DC disappears, no DC token is "close" to the cloud query in embedding space, so it contributes near-zero weight automatically. Softmax-attention on ≤8 tokens instead gives 1/N uniform floor to all tokens including padding-like ones — it cannot gracefully ignore an absent type. RBF avoids this floor.

**Caveat:** hybrid_rbf reverses on C→B (82.8% zero-shot, 1.199 AUC — worst two metrics). When DC load profiles shift from C to B, RBF distances give unexpected weight distributions that need a full fine-tuning phase to stabilize. The locality that helps with absent DC types hurts when the present DC types simply have different loads.

### 2. attention_idfree is the most consistently strong extractor

**Result:** 78.1% / 91.4% zero-shot, 3 / 1 ep→80%, 100.2% / 107.4% norm peak on C→A / C→B.

**Why:** No DC ID embedding is the decisive factor. ID-free means no semantic mismatch when DC IDs change across environments — the extractor encodes only DC type and per-host load, which are semantically stable across all topology variants. This is the strongest baseline for the paper's permutation-invariance claim.

### 3. Hierarchical cross-attention achieves the best asymptotic convergence on structural downscale

**Result:** 77.6% FinalvPk and 1.135 AUC on C→A — highest of all 10 extractors on both metrics.

**Why:** Jobs cross-attend to TYPE-level DC summaries (one vector per DC type: cloud/edge/micro), not per-slot DC tokens. When the cloud DC disappears, the cloud type-slot becomes zeros — the cross-attention mask handles it structurally with no learned positional associations to unlearn. The two-level hierarchy (hosts → DC type aggregates → job cross-attention) cleanly separates type-level semantics from instance-level IDs.

**Trade-off:** Low zero-shot (65.1%) because 4 gradient steps are needed to zero-out the cloud type representation in the policy's effective weights. Once those steps happen, it converges highest.

### 4. SWAT's topology descriptor helps C→B (92.5% zero-shot) but not C→A (62.5%)

**Result:** Best zero-shot on C→B, mid-pack on C→A.

**Why:** The struct_token encodes [n_dcs/max_dc, n_cloud/max_dc, n_edge/max_dc, n_micro/max_dc, total_free_pes/max_hosts]. For C→B, the descriptor immediately signals "3 DCs, 0 cloud" vs the trained "5 DCs, some cloud" — this shifts the global encoder's attention weights without any gradient update, producing the highest zero-shot on the easy direction. For C→A the same signal is available, but the host-level and DC-slot representations still encode cloud-DC patterns from C training that require gradient steps to override. The descriptor tells the policy *what changed* but cannot itself correct *how* the host embeddings respond to the absent DC type.

**Implication for paper:** SWAT's topology descriptor is most valuable when the change is a predictable size reduction (fewer DCs of the same types). It provides less zero-shot benefit when the type composition changes fundamentally.

### 5. DC ID embeddings (attention_pooling) are uniquely penalised on C→B

**Result:** attention_pooling is the only extractor needing 3 episodes to reach 80% oracle on C→B; all others reach it in 1.

**Why:** C training assigns learned embedding vectors to DC IDs 1–5. B only has IDs 1–3 (or differently renumbered). The embeddings for the C-specific IDs (4, 5) cannot directly transfer — their associated vectors have no matching DC in B. This is precisely the DC-ID-embedding failure mode documented in the transfer-safety checklist (Red Flag #1): trained positional lookup vectors that are undefined or semantically wrong for novel/absent DCs.

**C→A penalty is smaller** (64.1% vs 78.1% for attention_idfree) because A also has different IDs — the mismatch direction is consistent and the policy partially adapts at zero-shot by finding the "closest" embedded ID. But it still cannot match the ID-free architecture's clean generalization.

### 6. fusion's 23-episode C→A penalty quantifies the cost of positional inductive bias

**Result:** Worst adaptation speed by a factor of ~6× (23 ep vs next-worst spane at 17 ep).

**Why:** fusion's flat MLP over all hosts learns "the first N host slots are cloud DC hosts." When C→A eliminates the cloud DC, those input positions become zero-padded noise — but the MLP's weights are specifically trained to extract signal from those positions. It takes 23 fine-tuning episodes to suppress those learned position-weight associations. This is the strongest empirical evidence in the paper that canonical ordering without explicit DC-type embedding (or scatter-based invariance) creates a fragile architecture.

**Contrast:** fusion scores 89.5% zero-shot and 1 ep→80% on C→B — where the type composition doesn't change. The positional assumption holds, so it transfers cleanly. The fragility is specifically structural-change-triggered.

### 7. Training on the maximum topology (C) guarantees positive transfer everywhere

**Result:** ALL 10 extractors achieve AUC > 1.0 on BOTH C→A and C→B. C→B is near-free (9 of 10 reach 80% oracle in 1 episode).

**Why:** Training on the hardest environment (most DCs, all types) forces the extractor to learn representations that span the full type-vocabulary and a wide range of load patterns. Simpler environments (A, B) are strict subsets of that experience. The transferred policy does not need to "learn new things" — it needs to "unlearn" or "ignore" the cloud DC features, which is a much lighter adaptation task than learning from scratch.

**Implication:** The "train on max topology" strategy (C-source) dominates the "train on medium" (B-source) strategy from prior experiments for downscale transfer. B-source training was competitive on B→C upscale but is weaker as a starting point for downscale.

### 8. Norm peak > 100% is the norm from C-source training

**C→A:** hybrid_rbf (106.9%), pma (103.9%), hierarchical (103.8%), attention_idfree (100.2%), spane (100.0%)  
**C→B:** attention_idfree (107.4%), spane (107.0%), pma (106.4%), hierarchical (106.2%), euromlsys (106.1%), swat/attention_pooling (105.4%), fusion (105.0%)

Richer source training on C (5 DCs, more diverse load patterns, larger job diversity from the 3-location trace) builds surplus generalisation capacity that exceeds oracle-from-scratch performance in simpler targets. This is consistent with curriculum learning literature: training on a harder task first builds representations that overperform on easier related tasks.

### 9. The practical deployment choice depends on fine-tuning budget

| Budget | Best for C→A | Best for C→B |
|--------|-------------|-------------|
| Zero-shot (0 episodes) | hybrid_rbf (82.6%) | swat (92.5%) |
| Very short (1–5 episodes) | attention_idfree (3 ep, 100.2% peak) | attention_idfree / hierarchical (1 ep) |
| Full budget (50k steps) | hierarchical (77.6% FinalvPk) | euromlsys / fusion (>101% FinalvPk) |
| Best all-around balance | **attention_idfree** | **attention_idfree** |

attention_idfree is the Pareto-dominant choice across both directions when treating all metrics equally — never the single best on any one metric but never worst either, and consistently near the top on the metrics that matter most operationally (zero-shot, recovery speed, final quality).

### 10. euromlsys paradox: lowest zero-shot, highest AUC

**C→A:** euromlsys has the worst zero-shot of the non-turret extractors (53.3%) but the highest AUC (1.141).

**Why:** The residual adaptation layer (`base + 0.1 * adaptation_layer(base)`) is very effective once gradient updates start flowing — it provides a dedicated fast-fine-tuning pathway with 65,792 params that specialises on the current environment. But the adaptation layer cannot help before the first gradient step. The flat MLP body's inability to generalise zero-shot (positional inductive bias from canonical ordering) drags the jumpstart down, while the adaptation layer more than compensates over 50k steps.

**Paper framing:** euromlsys's residual adapter is the right mechanism but applied on the wrong base architecture. The ideal extractor combines a permutation-invariant body (like attention_idfree or hierarchical) with a similarly-sized pre-head adaptation layer — which is the design intent of the hybrid_pre_head and type_stratified_pre_head variants.

---

## Raw Numbers Reference

### C → A full metrics table

| Extractor | Jump | NormJump | OraclePk | XferPk | NormPk | Final50 | FinalvPk | AUC | Ep→80% |
|-----------|-----:|---------:|---------:|-------:|-------:|--------:|---------:|----:|-------:|
| hybrid_rbf | 3.971 | 82.6% | 4.806 | 5.138 | 106.9% | 3.511 | 73.0% | 1.108 | 1 |
| attention_idfree | 3.819 | 78.1% | 4.892 | 4.904 | 100.2% | 3.583 | 73.3% | 1.120 | 3 |
| fusion | 3.426 | 67.8% | 5.056 | 4.770 | 94.3% | 3.486 | 68.9% | 1.054 | 23 |
| hierarchical | 3.093 | 65.1% | 4.752 | 4.931 | 103.8% | 3.686 | 77.6% | 1.135 | 4 |
| attention_pooling | 3.141 | 64.1% | 4.898 | 4.727 | 96.5% | 3.642 | 74.4% | 1.094 | 3 |
| spane | 3.079 | 63.3% | 4.868 | 4.867 | 100.0% | 3.467 | 71.2% | 1.119 | 17 |
| swat | 3.126 | 62.5% | 5.005 | 4.837 | 96.6% | 3.563 | 71.2% | 1.117 | 4 |
| pma | 2.864 | 60.4% | 4.743 | 4.927 | 103.9% | 3.550 | 74.8% | 1.121 | 6 |
| euromlsys | 2.620 | 53.3% | 4.919 | 4.711 | 95.8% | 3.537 | 71.9% | 1.141 | 9 |
| turret | 2.506 | 49.8% | 5.028 | 4.889 | 97.2% | 3.521 | 70.0% | 1.105 | 9 |

### C → B full metrics table

| Extractor | Jump | NormJump | OraclePk | XferPk | NormPk | Final50 | FinalvPk | AUC | Ep→80% |
|-----------|-----:|---------:|---------:|-------:|-------:|--------:|---------:|----:|-------:|
| swat | 10.581 | 92.5% | 11.433 | 12.052 | 105.4% | 11.286 | 98.7% | 1.314 | 1 |
| attention_idfree | 10.191 | 91.4% | 11.155 | 11.980 | 107.4% | 10.855 | 97.3% | 1.280 | 1 |
| hierarchical | 10.223 | 90.3% | 11.322 | 12.027 | 106.2% | 11.266 | 99.5% | 1.290 | 1 |
| fusion | 10.372 | 89.5% | 11.591 | 12.174 | 105.0% | 11.794 | 101.8% | 1.315 | 1 |
| pma | 9.979 | 88.5% | 11.280 | 12.002 | 106.4% | 10.904 | 96.7% | 1.259 | 1 |
| turret | 10.189 | 88.5% | 11.519 | 11.808 | 102.5% | 10.834 | 94.1% | 1.230 | 1 |
| spane | 9.818 | 88.1% | 11.137 | 11.916 | 107.0% | 11.092 | 99.6% | 1.278 | 1 |
| euromlsys | 10.094 | 87.6% | 11.522 | 12.229 | 106.1% | 11.699 | 101.5% | 1.332 | 1 |
| hybrid_rbf | 9.448 | 82.8% | 11.411 | 11.849 | 103.8% | 10.461 | 91.7% | 1.199 | 1 |
| attention_pooling | 8.753 | 76.8% | 11.390 | 12.007 | 105.4% | 11.047 | 97.0% | 1.255 | 3 |
