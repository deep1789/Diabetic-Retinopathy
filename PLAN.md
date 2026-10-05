# Q1 Journal Plan: Unified DR Grading + Lesion Segmentation + Lesion Detection

Target: Elsevier, ≥14,000 words (main text, excl. references), result-oriented.
Status of numbers in this file: dataset statistics marked **(verify)** come from my recollection of the DDR paper and must be re-checked against the downloaded data. No results exist yet; none will be written until they are produced by real runs.

---

## 0. Data access (checked this session)

| Source | Reachable? | What it contains | Role in paper |
|---|---|---|---|
| `nkicsl/DDR-dataset` (GitHub) | Repo cloned. It holds **only README + LICENSE**. Data is on Baidu / Google Drive (10 zip chunks). The Drive folder page loads (HTTP 200); I have not pulled the chunks yet. | OIA-DDR: ~13.7k fundus images, 6-class grading (0–4 + ungradable), ~757 images with pixel masks (MA, HE, EX, SE) and boxes **(verify)** | **Primary** dataset for all three tasks |
| Kaggle `sovitrath/diabetic-retinopathy-224x224-gaussian-filtered` | **Downloaded anonymously** (448 MB, 3,664 PNG). | APTOS-2019 train only, 224×224, Gaussian-filtered. Counts: No_DR 1805, Mild 370, Moderate 999, Severe 193, Proliferate 295. **Grade labels only, no lesion labels.** | External grading test set / domain-shift study only |
| Other (similar) sets to add | Not yet attempted | IDRiD (grading + lesion seg + OD/fovea), FGADR (1,842 px-level), e-ophtha (MA/EX), DIARETDB1, Messidor-2, EyePACS/APTOS (grading) | External validation + lesion transfer |

Caveats that shape the design:
- 224×224 destroys microaneurysms (a few pixels at native size). The Kaggle set is therefore **not usable for segmentation/detection** and is a weak grading benchmark. Use native-resolution APTOS/EyePACS if the paper needs a grading external set.
- DDR licence is **CC BY-NC-SA 4.0**: cite Li et al., Information Sciences 501 (2019); derived data must be share-alike; non-commercial only.
- This container has **no GPU** (4 CPU, 15 GB RAM). I can build the pipeline, EDA, sample figures and smoke-test on CPU. Full training needs a GPU (Kaggle/Colab/Lab machine). Tell me where you will train.

---

## 1. Gap analysis (what the literature does, what is missing)

Most DR papers do one task. Multi-task work exists, but typically: (i) shares an encoder and sums losses, with no explicit link from lesion evidence to grade; (ii) uses lesion-annotated data only (small), or grade only; (iii) ignores that DR grade is an ordinal, clinically rule-defined function of lesion type/count/location; (iv) reports in-domain accuracy with no calibration or cross-dataset test; (v) detection and segmentation are treated as separate pipelines.

Gaps to exploit:
1. **G1** Grade is predicted *independently* of lesion predictions, so outputs can contradict (e.g. "Severe" with no predicted haemorrhage).
2. **G2** Lesion labels cover ≈5% of images; the other ~95% (image-level grade only) are wasted for lesion learning.
3. **G3** Small-lesion (MA) recall is poor, and detection is usually a separate network.
4. **G4** Ordinal label noise (inter-grader disagreement, adjacent-grade confusion) is mishandled by plain cross-entropy.
5. **G5** No coverage-guaranteed referral decision; calibration rarely reported.

> Before committing to novelty claims, run a systematic literature search (Scopus/WoS/Google Scholar, 2019–2026, queries: "diabetic retinopathy multi-task grading lesion segmentation", "lesion-guided grading", "DDR dataset", "clinical rule consistency"). I will produce a comparison table of ≥25 prior works. Novelty is only claimed for what that table supports.

---

## 2. Proposed methodology: **LesionRule-Net** (working name)

One shared encoder, three coupled heads, trained with grade-labelled and lesion-labelled data jointly.

### 2.1 Architecture
- **Encoder**: hierarchical ViT/ConvNeXt (e.g. Swin-T or ConvNeXt-T, optionally initialised from a retinal foundation model such as RETFound) at 1024×1024 (DDR native-ish) with a stride-4 high-resolution branch for MA.
- **Seg head**: FPN/UNet-style decoder → 4 lesion probability maps `P ∈ [0,1]^{4×H×W}` (MA, HE, EX, SE) + vessel/OD context channel.
- **Det head**: anchor-free (FCOS/CenterNet-style) on decoder features; boxes seeded by `P` peaks (Seg-guided proposals) to boost small-lesion recall at fixed FP/image.
- **Grade head** with **Lesion-Evidence Tokens (LET)**: zone-pooled lesion statistics (see 2.2) are projected to tokens that the grade query cross-attends to, alongside global image tokens.

### 2.2 Novel components (candidate contributions)
- **C1 Lesion-to-Grade Evidence Bottleneck.** Differentiable zone pooling (quadrants + macular-centred ring, ETDRS-inspired) of `P` gives soft counts `n_{c,z} = Σ_{x∈z} σ((P_c(x)-τ)/T)` (connected-component-free, soft). Grade logits depend on these through cross-attention, making grade explanations lesion-grounded.
- **C2 Clinical Rule-Consistency Loss.** Encode ICDR rules as soft logic with product t-norm, e.g. Severe ⇐ (HE-count ≥ 20 in 4 quadrants) ∨ (venous beading 2 quadrants) ∨ IRMA; Mild ⇐ MA present ∧ ¬(HE∨EX∨SE). Penalise `max(0, P(grade ≥ k) − Rule_k(n))`. Needs no extra labels.
- **C3 Grade→Lesion Weak Supervision with uncertainty gating.** For the ~95% unlabelled-lesion images, use grade as weak lesion evidence: image-level MIL loss (log-sum-exp pooling of `P`) + teacher–student pseudo-masks accepted only when epistemic uncertainty (MC-dropout/ensemble variance) < γ.
- **C4 Small-lesion-aware segmentation loss**: Focal-Tversky (β>α for recall) + boundary term + size-weighted class balancing; ablate vs Dice/BCE.
- **C5 Ordinal + noise-robust grading**: CORN ordinal head with Gaussian-smoothed soft labels over grade axis (σ learned) to model adjacent-grade disagreement.
- **C6 Conformal referral**: split-conformal prediction sets over ordinal grades giving finite-sample coverage ≥ 1−α, with "refer if set contains ≥ Moderate". Reports coverage/efficiency, ECE, selective-risk curves.

Contribution C1+C2+C3 is the core novelty; C4–C6 are supporting. Each must be ablated so claims are evidence-backed. If the lit review shows C2 or C3 already published, I will pivot the framing before writing.

### 2.3 Mathematical formulation (to appear in Section 3)
- Total loss: `L = L_cls + λ_s L_seg + λ_d L_det + λ_r L_rule + λ_w L_mil + λ_c L_cons`
- Ordinal (CORN): `L_cls = −Σ_{k=1}^{K-1} 1[y≥k] log p(y≥k | y≥k−1)` conditioned subsets.
- Soft-label smoothing: `q_j ∝ exp(−(j−y)²/2σ²)`, `L = KL(q‖p)`.
- Focal-Tversky: `FTL = (1 − TP/(TP + αFN + βFP))^{γ}`.
- MIL pooling: `s_c = (1/r) log( (1/|Ω|) Σ_x exp(r P_c(x)) )`.
- Rule loss with product t-norm, and proof sketch that the loss is zero iff the predictions satisfy the rule (Appendix).
- Conformal: nonconformity `s = 1 − p̂_y`, threshold `q̂ = Quantile_{⌈(n+1)(1−α)⌉/n}`; prediction set `{k: 1−p̂_k ≤ q̂}`; Theorem: `P(y∈C(x)) ≥ 1−α` (exchangeability).

### 2.4 Algorithms (pseudocode boxes)
1. Alg. 1: Joint training with mixed-supervision batches (labelled-lesion / grade-only sampler).
2. Alg. 2: Zone-wise soft lesion counting and rule-consistency evaluation.
3. Alg. 3: Uncertainty-gated pseudo-mask generation.
4. Alg. 4: Conformal calibration and inference for referral.
5. Alg. 5: Seg-guided proposal generation for detection.

---

## 3. Experimental design

**Data protocol**
- DDR official split for grading (train/val/test) and lesion tasks; patient-level splitting verified, duplicates checked via perceptual hash.
- Preprocessing: circular-crop, Ben-Graham/Gaussian local-contrast normalisation (the filtering used in the Kaggle set), CLAHE as ablation.
- External: APTOS/EyePACS/Messidor-2 (grading), IDRiD + FGADR (lesion seg, zero-shot and fine-tuned).
- Ungradable class: separate quality gate; reported with and without.

**Baselines**
- Grading: ResNet50, EfficientNet-B4, ConvNeXt-T, Swin-T, ViT-B, RETFound, CANet, DR multi-task prior works, DDR benchmark methods.
- Segmentation: U-Net, U-Net++, Attention U-Net, DeepLabV3+, TransUNet, Swin-UNet, SegFormer, SAM-fine-tuned, FGADR/DDR-reported baselines (HED, L-Seg, DeepLab).
- Detection: Faster R-CNN, RetinaNet, FCOS, YOLOv8/11, DETR/RT-DETR, plus DDR-reported.

**Metrics**
- Grading: QWK, accuracy, macro-F1, per-class sens/spec, AUC (referable DR), ECE, Brier.
- Segmentation: Dice, IoU, AUPR (preferred for tiny classes), per-lesion sensitivity.
- Detection: mAP@0.5 / @[.5:.95], AP-small, FROC sensitivity at 1/2/4/8 FP per image.
- Efficiency: params, FLOPs, latency, memory.
- Statistics: 5 seeds, mean±SD, 95% bootstrap CIs, DeLong (AUC), McNemar (accuracy), paired Wilcoxon (Dice), Holm correction.

**Ablations**: remove C1…C6 one at a time; zone definition; τ/T; λ sweep; encoder (CNN vs ViT vs foundation); resolution (224/512/1024); supervision ratio (1%…100% of lesion masks); preprocessing.

**Analyses**: rule-consistency rate (fraction of predictions violating ICDR rules, baseline vs ours); explanation fidelity (lesion deletion test); error analysis on adjacent-grade confusion; failure cases; cross-dataset drop; calibration; conformal coverage; clinical referral operating points.

---

## 4. Figures and tables

**Figures** (publication quality, vector, colour-blind-safe, shared style via the dataviz skill)
1. Graphical abstract.
2. Overall LesionRule-Net architecture.
3. Lesion-Evidence Bottleneck and zone pooling schematic.
4. DDR sample images per grade (0–4 + ungradable), with MA/HE/EX/SE overlays and boxes (**real images from the dataset**).
5. Kaggle Gaussian-filtered samples per grade vs original, preprocessing comparison.
6. Dataset statistics: class distribution, lesion-area histograms, lesion size/count per grade, brightness/quality.
7. Training curves and loss-component behaviour.
8. Confusion matrices (grade), ROC/PR curves.
9. Qualitative segmentation comparison (input, GT, baselines, ours) with error maps.
10. Qualitative detection comparison + FROC curves.
11. Lesion-grounded explanations vs Grad-CAM.
12. Reliability diagrams and conformal coverage/set-size plots.
13. Ablation bars / radar; supervision-ratio curve.
14. t-SNE/UMAP of grade embeddings (with vs without C1/C2).
15. Failure-case panel.
16. Cross-dataset generalisation plot.

**Tables**
1. Literature comparison (task, data, method, metrics, gaps).
2. Datasets summary (counts, resolution, annotations, licence).
3. Hyperparameters / implementation details.
4. Grading results (DDR) vs baselines.
5. Segmentation results per lesion type.
6. Detection results (mAP, FROC).
7. External validation (APTOS/IDRiD/Messidor-2/FGADR).
8. Ablation of components.
9. Rule-consistency violation rates.
10. Calibration and conformal results.
11. Computational cost.
12. Statistical tests.
13. Notation table.

---

## 5. Manuscript structure and word budget (Elsevier, ≥14,000)

| Section | Words |
|---|---|
| Highlights (3–5 bullets), graphical abstract, structured abstract (≤250), keywords | 350 |
| 1 Introduction (clinical burden, tasks, gaps, contributions list) | 1,400 |
| 2 Related work (grading, segmentation, detection, multi-task, rule/knowledge-guided, uncertainty) | 2,200 |
| 3 Materials: datasets, annotation, preprocessing, splits | 1,500 |
| 4 Methodology (architecture, C1–C6, maths, algorithms, complexity) | 3,600 |
| 5 Experimental setup (baselines, metrics, implementation, stats) | 1,300 |
| 6 Results (grading, seg, det, external, ablation, calibration, rule-consistency, efficiency) | 3,000 |
| 7 Discussion (interpretation, clinical implications, comparison, failure analysis, limitations, ethics) | 1,800 |
| 8 Conclusion & future work | 400 |
| Declarations (CRediT, data/code availability, competing interests, funding, AI-use statement) | 250 |
| **Total (excl. references and captions)** | **≈15,800** |
| References (~80–110, Elsevier numbered/CAS style) | — |

Format: Elsevier `elsarticle`/`cas-dc` LaTeX template, BibTeX, vector figures (PDF/SVG), supplementary material (extra qualitative results, proofs, hyperparameter search).

**Candidate journals** (check scope and current impact quartile before submission): Medical Image Analysis, Computers in Biology and Medicine, Artificial Intelligence in Medicine, Computer Methods and Programs in Biomedicine, Biomedical Signal Processing and Control, Pattern Recognition, Information Sciences (the DDR dataset was published there), Expert Systems with Applications.

---

## 6. Work plan (≈12 weeks)

| Weeks | Deliverable |
|---|---|
| 1 | Pull DDR chunks (gdown/Drive), verify counts and splits; EDA; sample figures (Fig. 4–6); systematic lit review table |
| 2 | Data pipeline, preprocessing, dataloaders, mixed-supervision sampler; baselines running (grading) |
| 3–4 | Baselines for seg/det; implement LesionRule-Net (C1, C4, C5) |
| 5–6 | Add C2, C3, C6; hyperparameter search; debug on subsets |
| 7–8 | Full runs, 5 seeds, external validation |
| 9 | Ablations, calibration, rule-consistency, explanation analyses |
| 10 | All figures/tables finalised from real outputs |
| 11 | Full manuscript draft; proofs in appendix |
| 12 | Internal review, language, reproducibility package (code, configs, weights), submission |

Risk register: GPU availability (blocking for week 4+); DDR lesion set small → high variance (mitigated by seeds, CIs, C3); C2 rule thresholds not directly observable (MA/HE counts only approximate ETDRS) → use soft rules and report as clinically *inspired*; novelty overlap with prior work (mitigated by the week-1 literature table).

---

## 7. What I can do next

1. Pull DDR from Google Drive and generate the real sample/statistics figures (Figs. 4–6).
2. Build the lit-review comparison table via web search.
3. Scaffold the code repo (configs, datasets, models, losses, training, evaluation) with CPU smoke tests.
4. Create the Elsevier LaTeX skeleton with all sections, notation, algorithm boxes and table stubs.

Open questions for you: Where will training run (GPU)? Preferred target journal? Any prior work/co-authors' constraints on the method?
