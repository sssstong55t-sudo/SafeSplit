# Energy-Guided Compact SafeSplit

## Project Summary

This project studies **SafeSplit**, a defense mechanism against **client-side backdoor attacks in U-shaped Split Learning (SL)**, and extends it toward a more efficient version called **Energy-Guided Compact SafeSplit**.

The project has two main parts:

1. **Reproduction**: reproduce the main SafeSplit pipeline, including the no-defense backdoor baseline and the full SafeSplit defense.
2. **Extension**: reduce SafeSplit's defense-side analysis cost by automatically selecting informative backbone layers and sampling channels, while preserving backdoor mitigation performance.

The core research question is:

> **Can SafeSplit maintain similar backdoor mitigation performance while analyzing only a compact, automatically selected subset of server-side backbone updates?**

---

## Background

### Split Learning Setting

In U-shaped Split Learning, the model is divided into three parts:

- **Head**: stored and trained on the client side.
- **Backbone**: stored and trained on the server side.
- **Tail**: stored and trained on the client side.

The client keeps the raw data and labels locally. During training, the client sends intermediate activations to the server, the server processes them through the backbone, and the client completes the forward and backward pass using the tail.

This setting is useful for privacy-preserving and resource-constrained scenarios because the computation-heavy backbone can be outsourced to a server.

### Threat Model

This project focuses on **client-side backdoor attacks**. A malicious client can manipulate:

- local training data,
- labels,
- loss function,
- smashed data,
- gradients sent to the server.

The attack goal is to make the final model behave normally on clean inputs but predict an attacker-chosen target label when a trigger is present.

For example, with a pixel trigger on CIFAR-10:

- Clean image → correct class.
- Triggered image → target class chosen by the attacker.

---

## Original SafeSplit

SafeSplit is a server-side defense designed for U-shaped Split Learning. It detects suspicious client-induced changes by analyzing the **server-side backbone updates**.

SafeSplit combines two types of analysis:

### 1. Static Analysis: DCT Frequency Analysis

SafeSplit transforms backbone updates into the frequency domain using the Discrete Cosine Transform (DCT).

The intuition is:

- Backdoor injection usually introduces abnormal changes to the model.
- Low-frequency DCT components can capture important structural changes.
- Poisoned updates may therefore have different frequency-domain signatures from benign updates.

For a backbone update:

```text
ΔB_t = B_t - B_{t-1}
```

SafeSplit computes a low-frequency DCT signature:

```text
S_t = DCT_low(ΔB_t)
```

Then it compares the DCT signatures of recent backbone updates using Euclidean distance.

### 2. Dynamic Analysis: Rotational Distance

SafeSplit also measures how the direction of backbone parameters changes over time.

The intuition is:

- A benign client usually continues the normal training trajectory.
- A malicious client may push the model toward a conflicting objective.
- This can create abnormal directional or rotational changes in the backbone update.

The simplified idea is:

```text
θ_t = angular position of backbone parameters
ω_t = θ_t - θ_{t-1}
RD_t = ω_t / (2π)
```

This captures dynamic changes that may not be visible from static magnitude-based comparison alone.

### 3. Circular Rollback Mechanism

SafeSplit does not simply remove a suspicious client forever. Instead, it searches backward through recent checkpoints and selects the most recent backbone checkpoint that appears benign under both static and dynamic analysis.

This is important because Split Learning is sequential:

- Client `i+1` starts from the model produced by client `i`.
- If client `i` is malicious, later benign clients may unknowingly train on a poisoned model.
- Rollback prevents the poisoned checkpoint from spreading through the training chain.

---

## Current Reproduction Plan

The reproduction part is divided into two stages.

## Stage 1: No-Defense Baseline

### Goal

Show that the U-shaped Split Learning system is vulnerable to client-side backdoor attacks when no defense is applied.

### Expected Behavior

The no-defense baseline should show:

- **High BA / ASR**: the backdoor attack succeeds.
- **Reasonable MA**: clean accuracy remains acceptable, so the attack is stealthy.

### Metrics

| Metric | Meaning | Desired by attacker |
|---|---|---|
| MA | Main Task Accuracy on clean test data | Keep high |
| BA | Backdoor Accuracy on triggered test data | Keep high |
| ASR | Attack Success Rate on triggered non-target samples | Keep high |

### Implementation Components

The current implementation includes:

- CIFAR-10 loading and preprocessing.
- ResNet-18 split into head, backbone, and tail.
- Client partitioning with main-label non-IID data distribution.
- Malicious clients with poisoned local datasets.
- Pixel trigger injection.
- Sequential client training.
- Clean accuracy and backdoor accuracy evaluation.

### Important Configuration

A typical baseline setting is:

```text
dataset              = CIFAR-10
model                = split ResNet-18
clients              = 10
malicious clients    = 2
IID rate             = 0.8
target label         = 2
trigger size         = 4
poison fraction      = 0.75
defense              = False
rotation analysis    = False
```

---

## Stage 2: Full SafeSplit Defense

### Goal

Reproduce the core SafeSplit defense logic:

- DCT-based static analysis.
- Rotational-distance-based dynamic analysis.
- Majority-based benign checkpoint selection.
- Rollback to the latest benign checkpoint.

### Expected Behavior

Compared with the no-defense baseline, full SafeSplit should show:

- Much lower BA / ASR.
- Similar or slightly reduced MA.
- Checkpoint rollback when suspicious updates are detected.

### Defense Logic

For each newly trained client checkpoint:

1. Store the server-side backbone checkpoint.
2. Compute DCT low-frequency signatures for recent backbone updates.
3. Compute rotational features for recent backbone updates.
4. Build static and dynamic distance scores.
5. Select the majority group under both metrics.
6. Roll back to the latest checkpoint that belongs to both majority groups.
7. Continue training from that selected benign checkpoint.

### Implementation Components

The current code already contains the following core functions:

```text
dct_1d_torch
dct_2d_torch
lowfreq_block_2d
low_frequency_signature
rotational_feature
build_rotation_scores
select_latest_benign_checkpoint
evaluate_clean
evaluate_ba
evaluate_asr
```

---

# Extension: Energy-Guided Compact SafeSplit

## Motivation

The original SafeSplit performs DCT-based static analysis and rotational dynamic analysis over server-side backbone updates.

However, when the backbone is large, full analysis over all backbone layers and all channels can be expensive.

The original paper demonstrates that analyzing backbone updates is effective, but it does not specifically study whether a **compact automatically selected subset of backbone layers and channels** is sufficient for detection.

Therefore, this project proposes:

> **Energy-Guided Compact SafeSplit: an efficient SafeSplit variant that analyzes only the most informative parts of the server-side backbone update.**

---

## Stage 3: Energy-Guided Automatic Layer Selection

### Problem

Full SafeSplit analyzes the backbone update broadly. This may include many layers that contribute little to detecting malicious behavior.

For large models, this creates redundant defense-side computation.

### Main Idea

For each backbone layer, compute the update energy:

```text
E_l = ||ΔW_l||_2^2
```

where:

```text
ΔW_l = W_l^t - W_l^{t-1}
```

Then select the top-`k` layers with the largest update energy.

SafeSplit-style DCT and rotational analysis are then applied only to these selected layers.

### Why This Is Reasonable

Backdoor injection usually requires noticeable model changes because the attacker is trying to introduce behavior that conflicts with the clean task.

Since SafeSplit itself is based on the intuition that poisoned updates create abnormal static and dynamic changes in the backbone, layers with larger update energy may contain stronger detection signals.

### Algorithm

```text
Input:
    current backbone B_t
    previous backbone B_{t-1}
    layer budget k

For each backbone layer l:
    ΔW_l = W_l^t - W_l^{t-1}
    E_l = ||ΔW_l||_2^2

Select:
    L_top = top-k layers by E_l

Run:
    DCT static analysis only on L_top
    Rotational dynamic analysis only on L_top
```

### Expected Benefit

This stage makes the defense more adaptive:

- The defense does not manually choose layers.
- The selected layers depend on the actual update energy.
- Redundant low-information layers can be skipped.

### Expected Result

The expected result is:

- Similar BA / ASR reduction compared with full SafeSplit.
- Similar MA.
- Lower defense-side analysis overhead.

---

## Stage 4: Channel Sampling

### Problem

Even after selecting important layers, convolutional layers can still contain many channels and parameters.

Analyzing the full tensor of selected layers may still be expensive.

### Main Idea

Instead of analyzing the entire weight tensor, analyze only a subset of channels.

For a convolutional layer with shape:

```text
[out_channels, in_channels, kernel_h, kernel_w]
```

we can sample a subset of output channels:

```text
selected_channels ⊂ {1, ..., out_channels}
```

Then DCT and rotational analysis are performed only on the sampled channels.

### Possible Sampling Strategies

#### 1. Uniform Channel Sampling

Randomly sample a fixed percentage of channels.

Example:

```text
sampling_rate = 0.25
```

This means only 25% of channels are used for defense-side analysis.

#### 2. Energy-Based Channel Sampling

Compute channel-level update energy:

```text
E_c = ||ΔW_c||_2^2
```

Then select the top channels with the largest update energy.

#### 3. Hybrid Sampling

First select top-`k` layers by layer energy, then within those layers select top channels by channel energy.

This is the most aligned version with the extension:

```text
Layer selection → Channel selection → Compact SafeSplit analysis
```

### Expected Benefit

Channel sampling should reduce:

- DCT computation cost.
- Rotational feature computation cost.
- Pairwise distance computation cost.
- Memory needed for storing compact signatures.

### Expected Result

The expected result is:

- Sampling rate decreases analysis overhead.
- BA / ASR should remain low when enough informative channels are retained.
- MA should remain stable because the training process itself is unchanged.

---

## Stage 5: Cost-Security Trade-off

### Goal

Stage 5 evaluates the trade-off between defense strength and computational cost.

The goal is not only to show that the compact method works, but also to show **how much analysis can be removed before security starts to degrade**.

### Experimental Variables

| Variable | Meaning |
|---|---|
| `k` | number of selected backbone layers |
| `sampling_rate` | percentage of channels analyzed |
| `selection_strategy` | full, random, energy-guided, hybrid |
| `defense_mode` | no defense, full SafeSplit, compact SafeSplit |

### Evaluation Metrics

| Metric | Meaning | Desired Result |
|---|---|---|
| MA | clean main task accuracy | high |
| BA | backdoor accuracy | low |
| ASR | attack success rate | low |
| analysis time | time spent on DCT + rotational analysis | low |
| analyzed parameters | number or ratio of analyzed parameters | low |
| rollback count | number of detected suspicious updates | reasonable and interpretable |

### Main Comparison Table

| Method | Layer Budget | Channel Sampling Rate | MA ↑ | BA ↓ | ASR ↓ | Analysis Time ↓ | Analyzed Params ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| No Defense | N/A | N/A | TBD | high expected | high expected | 0 | 0 |
| Full SafeSplit | all | 100% | TBD | low expected | low expected | highest | 100% |
| Random Compact SafeSplit | k | r | TBD | TBD | TBD | lower | lower |
| Energy-Guided Compact SafeSplit | k | r | TBD | low expected | low expected | lower | lower |

### Pareto Frontier Analysis

The final analysis should identify configurations that achieve the best balance between:

- security strength,
- clean utility,
- computational overhead.

A configuration is strong if it has:

```text
low BA / ASR
high MA
low analysis cost
```

This can be shown using a trade-off curve:

```text
x-axis: analyzed parameter ratio or analysis time
y-axis: BA / ASR / MA
```

The best configurations form a Pareto frontier: no other configuration is clearly better in both security and cost.

---

# Experimental Roadmap

## Experiment 1: Reproduce No-Defense Baseline

### Purpose

Verify that the attack works.

### Run

```bash
python main.py --rounds 50 --clients 10 --malicious_clients 2 --defense false --use_rotation false
```

### Expected Observation

```text
MA: reasonably high
BA/ASR: high
```

This proves the attack is successful and stealthy.

---

## Experiment 2: Reproduce Full SafeSplit

### Purpose

Verify that the original defense logic works.

### Run

```bash
python main.py --rounds 50 --clients 10 --malicious_clients 2 --defense true --use_rotation true
```

### Expected Observation

```text
MA: close to no-defense baseline
BA/ASR: significantly reduced
```

This proves that DCT + rotational analysis + rollback is effective.

---

## Experiment 3: Layer Selection Only

### Purpose

Test whether only a subset of layers is enough.

### Variables

```text
top_k_layers ∈ {1, 2, 3, 4, all}
```

### Expected Observation

If the idea works:

```text
top-k energy layers ≈ full SafeSplit security
but with lower analysis cost
```

---

## Experiment 4: Channel Sampling Only

### Purpose

Test whether partial channel analysis is enough.

### Variables

```text
sampling_rate ∈ {0.1, 0.25, 0.5, 0.75, 1.0}
```

### Expected Observation

If the idea works:

```text
medium sampling rate keeps BA/ASR low
while reducing analysis time
```

---

## Experiment 5: Full Compact SafeSplit

### Purpose

Combine layer selection and channel sampling.

### Variables

```text
top_k_layers ∈ {1, 2, 3, all}
sampling_rate ∈ {0.1, 0.25, 0.5, 1.0}
```

### Expected Observation

The best setting should keep:

```text
BA / ASR close to full SafeSplit
MA close to full SafeSplit
analysis cost much lower than full SafeSplit
```

---

# Expected Contributions

This project can be framed as a practical extension of SafeSplit.

## Contribution 1: Reproduction

Reproduce the SafeSplit pipeline on CIFAR-10 with:

- no-defense baseline,
- pixel-trigger attack,
- full SafeSplit defense,
- MA / BA / ASR evaluation.

## Contribution 2: Energy-Guided Layer Selection

Introduce an automatic layer selection mechanism based on backbone update energy.

This improves the original method by avoiding manual layer selection.

## Contribution 3: Channel Sampling

Reduce analysis cost further by applying DCT and rotational analysis only to sampled channels.

## Contribution 4: Cost-Security Trade-off

Evaluate the relation between:

- analysis budget,
- backdoor mitigation performance,
- clean accuracy,
- computational overhead.

This makes the work more engineering-oriented and practical.

---

# Why This Extension Is Defensible

This extension is defensible because it does **not** change the original SafeSplit security logic.

It keeps:

- the same threat model,
- the same attack setting,
- the same DCT static analysis idea,
- the same rotational dynamic analysis idea,
- the same rollback mechanism,
- the same evaluation metrics.

It only changes **which subset of the backbone update is analyzed**.

Therefore, the extension is low-risk and easy to explain:

> We are not replacing SafeSplit. We are making SafeSplit more efficient by reducing redundant analysis.

---

# Possible Limitations

## 1. High-Energy Layers May Not Always Be the Most Security-Relevant

A malicious update may hide in lower-energy layers.

Therefore, energy-guided selection must be compared with:

- full SafeSplit,
- random layer selection,
- fixed layer selection.

## 2. Too Aggressive Sampling May Miss Backdoor Signals

If the sampling rate is too low, the selected subset may not contain enough malicious signal.

Therefore, the project should evaluate several sampling rates.

## 3. Results May Depend on Model Architecture

The compact method may behave differently for:

- small CNNs,
- ResNet-18,
- larger backbones.

For the current project, CIFAR-10 with ResNet-18 is a reasonable starting point.

## 4. Cost Measurement Must Be Clear

The project should clearly distinguish:

- training time,
- defense-side analysis time,
- inference time.

Since this extension only changes the defense analysis, the most relevant metric is:

```text
defense-side analysis overhead
```

not normal inference delay.

---

# What to Show in Presentation

## Part 1: Reproduction, about 40%

### Slide 1: Problem Setting

- U-shaped Split Learning.
- Client-side backdoor attack.
- Server only sees backbone updates.

### Slide 2: No-Defense Baseline

- Malicious clients inject pixel trigger.
- BA / ASR high.
- MA remains acceptable.

### Slide 3: SafeSplit Defense

- Static DCT analysis.
- Dynamic rotational analysis.
- Rollback to latest benign checkpoint.

### Slide 4: Reproduction Status

- Current implementation components.
- Current metrics.
- Current plots / expected outputs.

## Part 2: Extension, about 60%

### Slide 5: Motivation

- Full SafeSplit analyzes many backbone parameters.
- Large backbones make defense-side analysis expensive.
- Question: do we really need to analyze everything?

### Slide 6: Stage 3 Layer Selection

- Compute layer update energy.
- Select top-k layers.
- Run SafeSplit analysis only on selected layers.

### Slide 7: Stage 4 Channel Sampling

- Sample informative channels.
- Reduce DCT and rotational computation.
- Compare sampling rates.

### Slide 8: Stage 5 Trade-off

- Compare MA, BA, ASR, analysis time.
- Show cost-security table.
- Identify Pareto-optimal configurations.

### Slide 9: Expected Contribution

- Reproduced SafeSplit.
- Proposed compact adaptive analysis.
- Demonstrated cost-security trade-off.

---

# Current Implementation Checklist

## Already Implemented

- [x] Safe torchvision import workaround.
- [x] Reproducible random seed setup.
- [x] CIFAR-10 data loading.
- [x] Pixel trigger injection.
- [x] Poisoned subset construction.
- [x] Main-label non-IID partitioning.
- [x] Split ResNet-18 construction.
- [x] Sequential client training.
- [x] Clean MA evaluation.
- [x] BA / ASR evaluation.
- [x] DCT low-frequency signature extraction.
- [x] Rotational feature extraction.
- [x] Majority-based benign checkpoint selection.
- [x] Metrics CSV output.

## To Implement Next

- [ ] Layer-wise update energy computation.
- [ ] Top-k layer selection.
- [ ] Channel-level update energy computation.
- [ ] Channel sampling inside selected layers.
- [ ] Compact DCT signature extraction.
- [ ] Compact rotational feature extraction.
- [ ] Analysis time measurement.
- [ ] Analyzed parameter ratio measurement.
- [ ] Cost-security comparison table.
- [ ] Trade-off plots.

---

# Suggested Repository Structure

```text
SafeSplit/
├── README.md
├── main.py
├── configs/
│   ├── baseline.yaml
│   ├── safesplit.yaml
│   └── compact_safesplit.yaml
├── outputs/
│   ├── metrics.csv
│   ├── final_results.txt
│   └── plots/
├── safesplit/
│   ├── dct_analysis.py
│   ├── rotation_analysis.py
│   ├── rollback.py
│   └── compact_selection.py
└── docs/
    ├── project_idea.md
    └── presentation_outline.md
```

---

# Suggested Final Report Structure

```text
1. Introduction
2. Background
   2.1 Split Learning
   2.2 Client-side Backdoor Attacks
   2.3 SafeSplit
3. Reproduction
   3.1 No-defense Baseline
   3.2 Full SafeSplit
   3.3 Metrics
4. Proposed Extension
   4.1 Motivation
   4.2 Energy-guided Layer Selection
   4.3 Channel Sampling
   4.4 Cost-Security Trade-off
5. Experiments
   5.1 Setup
   5.2 Baseline Results
   5.3 Full SafeSplit Results
   5.4 Compact SafeSplit Results
6. Discussion
7. Limitations
8. Conclusion
```

---

# One-Sentence Project Description

> This project reproduces SafeSplit, a server-side defense against client-side backdoor attacks in U-shaped Split Learning, and extends it with energy-guided layer selection and channel sampling to reduce defense-side analysis overhead while preserving backdoor mitigation performance.

---

# Short Abstract

SafeSplit is a defense against client-side backdoor attacks in U-shaped Split Learning. It detects poisoned client updates by combining static frequency-domain analysis and dynamic rotational analysis over server-side backbone updates, then rolls back to the latest benign checkpoint to prevent poisoned models from propagating through sequential training. This project first reproduces the no-defense backdoor baseline and the full SafeSplit defense on CIFAR-10. Then, it proposes Energy-Guided Compact SafeSplit, an efficient extension that automatically selects high-energy backbone layers and samples channels before applying SafeSplit-style analysis. The goal is to reduce defense-side analysis overhead while maintaining low backdoor accuracy and stable clean accuracy. The final evaluation studies the cost-security trade-off between full analysis and compact analysis.
