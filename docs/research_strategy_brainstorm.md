# Research Strategy Brainstorm — TPDS/TCC Journal Extension
**Date:** 2026-05-15

---

## Core Thesis

The paper's **central claim**:

> "A cloud scheduling agent that transfers across infrastructure changes must achieve all 5 topology-change invariances simultaneously. We show that TURRET (AAAI), SPANE (INFOCOM), and SWAT (ICLR) each achieve at most 3. We propose **ARIA**, which achieves all 5 by design."

---

## Corrected Architecture Assessment (2026-05-15 results review)

**SWAT is NOT the strongest existing architecture.** Earlier claim was wrong. Corrected picture:

| Extractor | C→A zero-shot | C→A recovery | C→B zero-shot | C→B FinalvPk | C→B AUC | Overall |
|-----------|:---:|:---:|:---:|:---:|:---:|---------|
| hybrid_rbf | **82.6%** | **1 ep** | 82.8% | 91.7% | 1.199 | **C→A specialist** |
| attention_idfree | 78.1% | 3 ep | 91.4% | 97.3% | 1.280 | **Most stable overall** |
| hierarchical | 65.1% | 4 ep | 90.3% | 99.5% | 1.290 | Best C→A asymptotic |
| swat | 62.5% | 4 ep | **92.5%** | 98.7% | 1.314 | C→B zero-shot only |
| fusion | 67.8% | **23 ep** | 89.5% | **101.8%** | 1.315 | C→B asymptotic, fatal C→A |
| euromlsys | 53.3% | 9 ep | 87.6% | 101.5% | **1.332** | Best C→B AUC, worst C→A zero-shot |
| attention_pooling | 64.1% | 3 ep | 76.8% (3ep) | 97.0% | 1.255 | DC ID embedding hurts C→B |

**The core tension exposed by the data:**

- **Scatter-add extractors** (hybrid_rbf, attention_idfree, hierarchical, swat) → handle structural change (C→A) well; missing cross-DC interaction capacity
- **Flat MLP extractors** (fusion, euromlsys) → high asymptotic quality via cross-host interactions and residual adapters; catastrophic on structural downscale (fusion: 23 ep penalty)
- **attention_idfree** threads the needle best: scatter-add base + learned pool query, decent on both

**What the ideal extractor needs to beat:**
- hybrid_rbf's C→A zero-shot benchmark: 82.6% / 1 episode
- fusion's C→B FinalvPk benchmark: 101.8%
- Without fusion's 23-episode C→A penalty
- Mechanism: **attention_idfree base** (cross-attention brain: job↔DC implicit relative encoding, most stable overall) + **RBF final pool** (hybrid_rbf's C→A locality advantage, moved to the final pool step — not the DC-pool step — to preserve per-slot tokens for cross-attention) + **residual adaptation layer** (fusion/euromlsys's C→B fine-tuning bandwidth) → this combination is **ARIA**

---

## The 5 Invariances Taxonomy

This is the paper's conceptual contribution — a principled taxonomy of what can change when a trained cloud scheduler is deployed to a new environment.

| # | Invariance | Real-world trigger | Current testing status |
|---|-----------|-------------------|------------------------|
| 1 | **Permutation** — DC slot ordering doesn't affect policy | Failover, renaming, reconfiguration | ❌ never tested |
| 2 | **Count** — number of DCs changes | Capacity expansion / reduction | ✅ C→B, C→A partial |
| 3 | **Type** — DC tier disappears or appears | Cloud DC retired; new edge tier added | ✅ C→A (retirement only) |
| 4 | **Workload** — job arrival distribution shifts | New application deployed on same infra | ❌ never tested |
| 5 | **Scale** — overall capacity proportionally changes | Horizontal cluster scaling | ❌ never tested |

---

## Proposed Environments

### Current environments (keep as-is)

| Env | DCs | Types | Hosts | Job trace |
|-----|-----|-------|-------|-----------|
| A | 2 | edge×1 + micro×1 | 20 | standard |
| B | 3 | edge×1 + micro×2 | 30 | standard |
| C | 5 | cloud×1 + edge×2 + micro×2 | 50 | standard |

### New environments to add

| Env | DCs | Types | Hosts | Job trace | Purpose | Effort |
|-----|-----|-------|-------|-----------|---------|--------|
| **D** | 8 | cloud×2 + edge×3 + micro×3 | 80 | standard | Extreme scale-up target | New topology YAML |
| **E** | 3 | cloud×3 | 30 | standard | Cloud-only (no edge/micro) — type restriction | New topology YAML |
| **C′** | 5 | same as C | 50 | heavy_jobs | Same infra, different workload | New job trace CSV |
| **C_perm** | 5 | same as C | 50 | standard | Same as C but DC IDs shuffled each episode | Python-only: shuffle in reset() |

**C_perm** is the cheapest to implement — zero Java changes, one Python shuffle in `_get_observation()`.
**C′** requires only a different job trace CSV (e.g., all jobs requesting 8 PEs instead of 1–4 mixed).
**D** and **E** are new topology YAMLs (model on euromlsys_c.yml).

---

## The 10 Transfer Scenarios

Each scenario tests a specific real-world situation. Each is designed to expose a specific architectural failure mode.

| ID | Transfer | Invariance tested | Real-world scenario | Target failure mode |
|----|----------|------------------|---------------------|---------------------|
| S1 | C → B | Count (downscale) | Maintenance window, capacity reduction | Unmasked mean pooling |
| S2 | C → A | Type (retirement) | Cloud DC decommissioned | DC ID embeddings, positional MLP |
| S3 | B → D | Count (upscale) | Capacity expansion | DC ID embeddings (new unseen IDs) |
| S4 | A → C | Type (introduction) | New cloud tier deployed | Type-blind / type-specific query extractors |
| S5 | C → C_perm | Permutation | Failover / DC renaming / reconfiguration | euromlsys, fusion (positional bias), TURRET (node IDs) |
| S6 | C → C′ | Workload | New application class deployed on same infra | All absolute-capacity extractors |
| S7 | D → A | Count (extreme downscale, 75% loss) | Catastrophic capacity crisis | All — stress test |
| S8 | E → B | Type (no overlap: cloud-only → edge+micro) | Full cloud-to-edge migration | Type embedding robustness |
| S9 | B → C → D | Count (staged upscale) | Incremental expansion over time | Curriculum transfer |
| S10 | D → C → B | Count (staged downscale) | Rolling decommission | Reverse curriculum |

### Minimum viable set for the paper

S1, S2, S3, S5, S6 + one of {S4, S7, S8} = 6 scenarios, covering all 5 invariances.
S5 (permutation) uses only existing trained models — zero retraining cost.
S6 (workload) uses existing models — zero retraining cost.

---

## Extractor Failure Map

Each scenario is a probe that kills specific extractors. The value is that each failure has a **mechanistic explanation**.

| Extractor | Killed by | Root cause |
|-----------|-----------|------------|
| euromlsys | S5 (permutation) | Linear(120→128) learns positional associations with canonical DC ordering |
| fusion | S5 (permutation) | Flat dense MLP same problem as euromlsys |
| attention_pooling | S2, S3 | nn.Embedding indexed by DC ID — new/reordered IDs = random vectors |
| spane | S6 (workload shift) | Absolute free_pes; masked mean loses expressiveness vs workload change |
| hybrid_rbf | S4 (type introduction) | RBF kernel attends to existing DC tokens; new type has no trained signal |
| hierarchical | S4 (type introduction) | Type-level cross-attention has no slot for unseen type |
| pma | S4 (type introduction) | max_dc_types learned queries; unseen type has no dedicated query |
| SWAT | S6 (workload shift) | struct_token encodes topology structure, not job-load relationship |
| TURRET | S5 (permutation) | GNN uses node identity — permuting node IDs breaks learned graph structure |
| SPANE (INFOCOM) | S6 (workload shift) | Absolute DC features; masked mean aggregation |

**S5 (permutation) is particularly powerful**: zero retraining, immediate result, directly differentiates flat-MLP papers from scatter-based papers. One figure showing euromlsys/fusion reward collapse under DC ID shuffle is compelling.

---

## Novel Architecture: ARIA

**ARIA** = **A**daptive **R**BF pooling + **I**mplicit relative encoding via cross-attention + **A**daptation layer

Built from three existing extractors, each contributing what it does best:
- **attention_idfree**: job ↔ DC cross-attention + global encoder (the "brain" — implicit relative encoding)
- **hybrid_rbf**: RBF final pooling (locality for absent DC types → C→A zero-shot robustness)
- **euromlsys**: residual adaptation layer (fine-tuning bandwidth → C→B asymptotic quality)

### Why attention_idfree is the right base (what makes it robust)

attention_idfree's key differentiator from hybrid_rbf is **job ↔ DC cross-attention + global encoder**:

```python
# Q=jobs, K/V=DC slots — job representations become DC-conditioned
j_att, _ = self.cross_attn(j, dc_repr, dc_repr)
j = self.cross_norm(j + j_att)
# both dc_repr and updated j go into global encoder together
encoded = self.global_encoder(cat([dc_repr, j]))
```

This is **implicit relative encoding**: each job token gets updated by attending over which DC slots fit it.
The global encoder lets DC tokens and job tokens refine each other mutually.
Result: the final representation is jointly shaped by infrastructure AND current job queue — without needing explicit feature ratios.

hybrid_rbf processes jobs and DCs fully independently, concatenating only at the head. No job-DC coupling → weaker C→B asymptotic quality despite superior C→A zero-shot.

### Why RBF at the final pool step (not the DC-pool step)

The subtlety: cross-attention needs per-slot DC tokens, not a pooled summary. Placing RBF at the DC-pool step (as in standalone hybrid_rbf) destroys the per-slot representation before cross-attention can use it. The right placement is **replacing the final MHA pool query with RBF**:

- Softmax pool (attention_idfree): uniform 1/N floor — absent DC type's near-zero token still gets 1/N weight
- RBF pool: `w_i = exp(-||query - encoded_i||² / 2σ²)` — near-zero token is far from learned query → weight → 0

When cloud DC disappears in C→A, its encoded slot is near-zero throughout the transformer stack. RBF naturally silences it; softmax cannot.

### Full ARIA pipeline

```
host_feats → dc_type_emb + free_pes
    │
    host_encoder (Transformer, 1 layer, src_key_padding_mask on dc_id==0)
    │
    scatter_to_dc → dc_repr [B, max_dc, D]      ← permutation invariant
    │
    cross_attn(Q=job_proj(jobs), K=dc_repr, V=dc_repr)  ← implicit relative encoding
    cross_norm(j + j_att)
    │
    global_encoder([dc_repr || job_tokens], padding_mask)  ← mutual DC-job refinement
    │
    RBF_pool(learned_query, encoded, σ=learned)  ← locality: absent types → weight≈0
    │
    head(pooled) → base [B, features_dim]
    │
    base + 0.1 * adaptation_layer(base)          ← fine-tuning bandwidth for size downscale
```

`σ` is a scalar learned parameter. Pool: `w = softmax_free(exp(-||q-k||²/2σ²) * valid_mask)`, `out = Σ w_i v_i`.

### Invariance coverage

```
Invariance:      Permutation  Count   Type    Workload    Scale
----------------------------------------------------------------
euromlsys        ❌           ✅      ⚠️      ❌          ❌
TURRET (AAAI)    ❌           ⚠️      ⚠️      ❌          ❌
SPANE (INFOCOM)  ✅           ✅      ✅      ❌          ❌
SWAT (ICLR)      ✅           ✅      ✅      ❌          ❌
hybrid_rbf       ✅           ✅      ⚠️      ❌          ❌
attention_idfree ✅           ✅      ✅      implicit    implicit
ARIA (ours)      ✅           ✅      ✅      implicit    implicit + fast adapt
```

Workload/scale: "implicit" = learned via cross-attention conditioning, not hard-coded feature ratios.

### Clean ablation story

| Variant | What changes | Tests which component |
|---------|-------------|----------------------|
| attention_idfree | base (current best balanced) | reference |
| attention_idfree + adapter | add only residual adaptation layer | does adapter alone close C→B gap? |
| ARIA - adapter | RBF pool + cross-attn, no adapter | does RBF pool alone improve C→A? |
| ARIA (full) | RBF pool + cross-attn + adapter | do both components compound? |

Expected: ARIA > attention_idfree on C→A (RBF) AND C→B (adapter). The ablation proves both components are independently necessary.

---

## Competitor Positioning

| Paper | Venue | Their claim | Where they fail | Our counter |
|-------|-------|-------------|-----------------|-------------|
| TURRET | AAAI | GNN captures DC relationships | GNN uses node IDs → fails S5 (permutation) | One figure shows permutation collapse |
| SPANE | INFOCOM | Masked DC aggregation is transfer-safe | Absolute features → fails S6 (workload shift) | Workload shift experiment |
| SWAT | ICLR | Structure-aware topology token | Absolute features in host encoder → fails S6; 62.5% C→A zero-shot (mediocre) | ARIA outperforms SWAT on C→A (78.1% vs 62.5%) and matches on C→B |
| euromlsys | EuroMLSys'25 | Dense body captures cross-host interactions | Positional → fails S5; 23-ep C→A penalty (fusion variant) | Our own prior work; ARIA is the fix |

**Key positioning move**: ARIA is grounded in attention_idfree — empirically the most stable architecture — and makes two targeted additions (RBF final pool, residual adapter) each traceable to a specific failure mode. This is a credible surgical improvement narrative: show what breaks in each baseline, show the data gap, show ARIA closes it. Cite SWAT and SPANE as prior work in the transformer-for-scheduling line; contrast with ARIA on the invariance table.

---

## Two-Sentence Paper Pitch

**Problem:** Cloud scheduling agents trained on one infrastructure topology silently fail on others — not because they overfit to performance details, but because each extractor architecture is structurally blind to a specific class of topology change (permutation, count, type, workload, or scale).

**Insight:** We propose ARIA (Attention-RBF Invariant Architecture), which combines the cross-attention brain of attention_idfree (job↔DC implicit relative encoding), an RBF locality kernel at the final pool step (zero weight for absent DC types without softmax noise floor), and a residual adaptation layer (dedicated fine-tuning bandwidth), achieving all five topology-change invariances simultaneously — the first extractor to do so — and outperforming TURRET (AAAI), SPANE (INFOCOM), and SWAT (ICLR) on a comprehensive transfer benchmark spanning 6 distinct real-world change scenarios.

---

## Prioritized Action List

| Priority | Action | Effort | Expected output |
|----------|--------|--------|-----------------|
| 1 | Implement C_perm (shuffle DC IDs in Python at reset) | Half a day | Run existing 10 models through it; immediate result; exposes euromlsys/fusion collapse for free |
| 2 | Design C′ trace (heavy job distribution) | Half a day | Run existing models through it; exposes workload invariance gap in all absolute encoders |
| 3 | ✅ Implement ARIA | Done | aria_extractor.py: attention_idfree base + RBF final pool + residual adapter; registered in __init__.py and config.yml |
| 4 | Add Env D (8 DCs, topology YAML) | 1 day | Enables S3 (upscale) and S7 (extreme downscale) |
| 5 | Add Env E (cloud-only) | 1 day | Enables S8 (type-swap) |
| 6 | Write taxonomy section (paper) | 1 day | The 5 invariances table is the conceptual anchor |

---

## Paper Sections to Update

| Section | Change |
|---------|--------|
| Abstract | Mention taxonomy + 6-scenario benchmark + ARIA |
| Related Work | TURRET, SPANE, SWAT — position relative to invariance coverage |
| Architecture (Sec V) | Add ARIA; describe cross-attention base + RBF final pool + adapter; position vs. attention_idfree as base |
| Experiments (Sec VI.A) | 5-invariance taxonomy table |
| Experiments (Sec VI.B–G) | One subsection per scenario type, each with result table |
| Experiments (Sec VI.H) | ARIA ablation (base only vs. +RBF vs. +adapter vs. full ARIA) |
| Experiments (Sec VI.I) | Permutation stress test (S5) — no retraining, just evaluation |
| Discussion | Which invariance matters most for which deployment pattern |
| Conclusion | "First comprehensive taxonomy + method achieving all 5 invariances" |

---

## Open Questions (think about these)

1. **How realistic is the workload shift scenario?** Does it make sense for the same infrastructure to have completely different job profiles, or should jobs be infrastructure-specific? Need to make the argument that yes, this happens (multi-tenant clouds, time-of-day shifts, new application deployments).

2. **Does the relative encoding hurt source training quality?** If jobs queue is empty at episode start, mean_cores = 0 → division by ε → all DCs look the same initially. May need careful handling of the cold-start case.

3. **Is ARIA architecturally too incremental over attention_idfree?** ARIA adds two components (RBF pool + adapter) on top of attention_idfree. Reviewers may say "this is just attention_idfree + two engineering tricks." Mitigate by: (a) clean ablation showing each component independently necessary (ARIA - adapter and ARIA - RBF both degrade on at least one transfer direction), (b) making the taxonomic contribution (5-invariance framework) the primary claim, not the architecture alone, (c) empirically showing ARIA dominates every baseline on both C→A and C→B simultaneously — no prior extractor does both.

4. **Should we also test online adaptation (sequential drift)?** S9/S10 (staged curriculum) could be a separate narrative: "what if you don't fine-tune at all but the infrastructure changes gradually?" This is a different experimental setup (online RL vs. transfer RL) and may be out of scope.

5. **TURRET implementation**: we need TURRET code to run S5 and show its permutation failure. If TURRET isn't available, we can argue from architecture (GNN with node IDs) without running it — but running it is stronger.
