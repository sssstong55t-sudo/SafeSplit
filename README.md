# Energy-Guided Compact SafeSplit

This repository contains a course project for reproducing and extending **SafeSplit: A Novel Defense Against Client-Side Backdoor Attacks in Split Learning**.

The project has two main parts:

1. **Reproduction**: reproduce the main SafeSplit setting, including the no-defense backdoor baseline and the SafeSplit defense pipeline.
2. **Extension**: propose and evaluate **Energy-Guided Compact SafeSplit**, an efficiency-oriented variant that reduces defense-side analysis overhead by applying SafeSplit-style analysis only to a compact subset of server-side backbone updates.

> The extension does **not** change the attack setting, the U-shaped Split Learning training pipeline, the SafeSplit decision logic, or the rollback mechanism. It only changes which subset of server-side backbone parameters is used during static and dynamic defense analysis.

---

## 1. Background

### 1.1 Split Learning

Split Learning (SL) is a distributed training framework where a neural network is divided between clients and a server. In the **U-shaped Split Learning** setting, the model is split into three parts:

```text
Client-side Head  ->  Server-side Backbone  ->  Client-side Tail
```

The client keeps the input data, labels, head model, and tail model locally. The server only holds the backbone. During training, clients send intermediate representations, also called **smashed data**, to the server, and the server computes the backbone forward/backward pass.

This setting is useful for resource-constrained clients because the computationally heavy backbone is outsourced to the server. However, it also introduces a security problem: malicious clients may manipulate their local data, labels, loss, gradients, or smashed data to inject a backdoor into the shared model.

### 1.2 Backdoor Attacks in Split Learning

A backdoor attack aims to make the trained model behave normally on clean inputs while predicting an attacker-chosen target label when a trigger is present.

In this project, the main attack setting is a **client-side backdoor attack** in U-shaped Split Learning. The malicious clients may use a pixel trigger or a semantic trigger to poison the training process.

The key evaluation idea is:

- Clean inputs should still be classified correctly.
- Triggered inputs should be misclassified into the attack target label if the attack succeeds.

Therefore, a successful attack usually has:

```text
High Backdoor Accuracy (BA)
High or stable Main Task Accuracy (MA)
```

A successful defense should have:

```text
Low Backdoor Accuracy (BA)
Stable Main Task Accuracy (MA)
```

---

## 2. Original SafeSplit

SafeSplit is a defense against client-side backdoor attacks in Split Learning. It is deployed on the server and analyzes the server-side backbone updates.

The original SafeSplit pipeline contains three important ideas:

### 2.1 Static Frequency-Domain Analysis

SafeSplit applies a Discrete Cosine Transform (DCT) to backbone updates and compares low-frequency representations. The intuition is that backdoor training introduces abnormal changes in the model update, and these changes can be detected in the frequency domain.

### 2.2 Dynamic Rotational Analysis

SafeSplit also computes a rotational distance metric to capture the direction and trajectory of backbone parameter changes during training. This complements the static DCT analysis.

### 2.3 Rollback Mechanism

Because Split Learning is sequential, a malicious client can poison the model state used by later benign clients. SafeSplit therefore uses a rollback mechanism to revert to the most recent checkpoint considered benign.

The simplified logic is:

```text
After a client finishes training:
    1. Store the current server-side backbone checkpoint.
    2. Compare recent backbone updates using DCT-based static analysis.
    3. Compare recent backbone updates using rotational dynamic analysis.
    4. Identify whether the current checkpoint belongs to the benign majority.
    5. If suspicious, roll back to the latest benign checkpoint.
```

---

## 3. Project Goal

The goal of this project is not to replace SafeSplit. Instead, the goal is to reproduce its core idea and then study whether the defense-side analysis can be made more compact.

The central research question is:

> Can SafeSplit maintain similar backdoor mitigation performance while analyzing only a compact, automatically selected subset of server-side backbone updates?

This leads to the proposed extension:

```text
Energy-Guided Compact SafeSplit
= Energy-guided automatic layer selection
+ Channel sampling
+ Cost-security trade-off analysis
```

---

## 4. Project Structure

The current repository contains:

```text
SafeSplit/
├── README.md
├── 2501.06650v2.pdf
└── reproduce baseline (4.27) .ipynb
```

Expected extended structure:

```text
SafeSplit/
├── README.md
├── 2501.06650v2.pdf
├── reproduce baseline (4.27) .ipynb
├── outputs/
│   ├── metrics.csv
│   ├── final_results.txt
│   └── plots/
│       ├── ma_curve.png
│       ├── ba_curve.png
│       └── cost_security_tradeoff.png
└── docs/
    └── presentation.pdf
```

If the project is later refactored into Python scripts, the recommended structure is:

```text
SafeSplit/
├── README.md
├── requirements.txt
├── main.py
├── configs/
│   ├── baseline.yaml
│   ├── safesplit.yaml
│   └── compact_safesplit.yaml
├── src/
│   ├── data.py
│   ├── models.py
│   ├── split_learning.py
│   ├── attacks.py
│   ├── safesplit.py
│   ├── compact_safesplit.py
│   └── evaluation.py
├── notebooks/
│   └── reproduce_baseline.ipynb
└── outputs/
```

---

## 5. Reproduction Plan

The reproduction part is divided into two stages.

### Stage 1: No-Defense Baseline

The first stage demonstrates that U-shaped Split Learning is vulnerable to client-side backdoor attacks.

In this setting:

- Defense is disabled.
- Malicious clients inject a backdoor using poisoned data or manipulated training behavior.
- The model is evaluated on both clean test data and triggered test data.

Expected behavior:

```text
MA remains reasonably high.
BA becomes very high.
```

This shows that the backdoor is effective while the normal task performance remains acceptable.

### Stage 2: Full SafeSplit Defense

The second stage reproduces the SafeSplit-style defense pipeline.

In this setting:

- Defense is enabled.
- Static DCT-based frequency analysis is applied to server-side backbone updates.
- Dynamic rotational analysis is applied to recent backbone updates.
- The rollback mechanism is used to skip suspicious checkpoints.

Expected behavior:

```text
BA decreases significantly.
MA remains close to the no-defense setting.
```

This shows whether the SafeSplit-style analysis can mitigate the backdoor without destroying main-task utility.

---

## 6. Extension: Energy-Guided Compact SafeSplit

The extension focuses on reducing the computational overhead of SafeSplit's server-side analysis.

Original SafeSplit analyzes server-side backbone updates. However, for large neural networks, the backbone may contain many layers and parameters. Running DCT and rotational analysis on all backbone parameters can be expensive.

The proposed extension asks whether it is necessary to analyze the full backbone, or whether a compact subset is enough.

---

## 7. Stage 3: Energy-Guided Automatic Layer Selection

### 7.1 Motivation

Backdoor injection usually requires modifying the model behavior. Since SafeSplit is based on the intuition that poisoned updates introduce abnormal static and dynamic changes, layers with larger update energy may contain stronger detection signals.

### 7.2 Layer Energy

For each backbone layer, compute the update energy:

```text
E_l = || ΔW_l ||_2^2
```

where:

- `l` is the layer index.
- `W_l` is the parameter tensor of layer `l`.
- `ΔW_l = W_l^t - W_l^{t-1}` is the layer update after a client trains.

### 7.3 Selection Rule

Select the top-k layers with the largest update energy:

```text
SelectedLayers = TopK(E_l)
```

Then run SafeSplit-style DCT and rotational analysis only on these selected layers.

### 7.4 Important Clarification

Energy-guided layer selection is only a lightweight heuristic. It does not directly classify a client as malicious. The actual defense decision is still made using the SafeSplit-style static and dynamic analysis.

This is important because a low-energy layer may still contain useful information. Therefore, the project should compare energy-guided selection with:

- full-backbone SafeSplit,
- random layer selection,
- fixed layer selection,
- different values of top-k.

---

## 8. Stage 4: Channel Sampling

### 8.1 Motivation

Even after selecting a few backbone layers, convolutional layers can still contain many channels and parameters. Full tensor analysis may still be expensive.

Therefore, the next step is to sample only a subset of channels inside the selected layers.

### 8.2 Sampling Strategy

For a convolutional weight tensor:

```text
W_l ∈ R^{C_out × C_in × K × K}
```

sample a subset of output channels:

```text
W_l[selected_channels, :, :, :]
```

Possible sampling strategies:

1. **Random channel sampling**
2. **Energy-based channel sampling**
3. **Fixed-ratio channel sampling**
4. **Top-k channel sampling**

### 8.3 Sampling Rates

The experiments can evaluate different sampling rates:

```text
100%   full SafeSplit baseline
75%    mild compression
50%    medium compression
25%    strong compression
10%    aggressive compression
```

The key question is whether the compact version can still keep BA low and MA stable.

---

## 9. Stage 5: Cost-Security Trade-off

The final extension stage evaluates the trade-off between defense strength and defense-side overhead.

### 9.1 Metrics

The main metrics are:

| Metric | Meaning | Desired Direction |
|---|---|---|
| MA | Main Task Accuracy on clean test data | Higher is better |
| BA | Backdoor Accuracy on triggered test data | Lower is better |
| ASR | Attack Success Rate | Lower is better |
| Defense analysis time | Time spent on DCT/RD analysis | Lower is better |
| Selected parameter ratio | Fraction of backbone parameters analyzed | Lower is better |
| Memory overhead | Extra memory used by defense analysis | Lower is better |
| Rollback count | Number of times the defense rolls back | Should be reasonable |

### 9.2 Recommended Result Table

| Method | Selected Layers | Channel Sampling Rate | Selected Parameter Ratio | MA ↑ | BA ↓ | Analysis Time ↓ |
|---|---:|---:|---:|---:|---:|---:|
| No Defense | N/A | N/A | 0% | TBD | TBD | 0 |
| Full SafeSplit | All | 100% | 100% | TBD | TBD | TBD |
| Compact SafeSplit | Top-2 | 50% | TBD | TBD | TBD | TBD |
| Compact SafeSplit | Top-2 | 25% | TBD | TBD | TBD | TBD |
| Compact SafeSplit | Top-1 | 25% | TBD | TBD | TBD | TBD |
| Random Sampling | Random | 25% | TBD | TBD | TBD | TBD |

### 9.3 Recommended Plot

The final report should include a cost-security trade-off plot:

```text
x-axis: selected parameter ratio or analysis time
 y-axis: BA / ASR / MA
```

A good result would show that compact SafeSplit achieves similar BA and MA to full SafeSplit while using fewer analyzed parameters or less defense-side analysis time.

---

## 10. Experimental Setup

The default experimental setting follows the paper-style CIFAR-10 split learning setup as closely as possible within the project implementation.

Recommended configuration:

| Parameter | Value |
|---|---:|
| Dataset | CIFAR-10 |
| Model | ResNet-18 style split model |
| Number of clients | 10 |
| Malicious clients | 2 |
| IID rate | 0.8 |
| Trigger type | Pixel trigger |
| Target label | 2 |
| Trigger size | 4 × 4 |
| Poison fraction | 0.75 |
| Defense baseline | Full SafeSplit-style DCT + rotational analysis |
| Extension | Energy-guided layer selection + channel sampling |

---

## 11. How to Run

### 11.1 Run in Google Colab

The easiest way to run the current project is through the notebook:

```text
reproduce baseline (4.27) .ipynb
```

Steps:

1. Open the notebook in Google Colab.
2. Set runtime to GPU if available.
3. Run all setup cells.
4. Run the no-defense baseline.
5. Run the SafeSplit defense setting.
6. Run the compact SafeSplit extension experiments.
7. Check the generated metrics and plots.

### 11.2 Run Locally

Clone the repository:

```bash
git clone https://github.com/sssstong55t-sudo/SafeSplit.git
cd SafeSplit
```

Create a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
```

Install dependencies:

```bash
pip install --upgrade pip
pip install torch torchvision numpy pandas matplotlib tqdm scipy
```

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
reproduce baseline (4.27) .ipynb
```

---

## 12. Output Files

The notebook/script should generate output files such as:

```text
outputs/
├── metrics.csv
├── final_results.txt
└── plots/
    ├── ma_curve.png
    ├── ba_curve.png
    ├── asr_curve.png
    └── cost_security_tradeoff.png
```

Recommended content of `metrics.csv`:

| Column | Meaning |
|---|---|
| round | Training round |
| client_id | Current client |
| is_malicious | Whether the client is malicious |
| defense_enabled | Whether defense is enabled |
| selected_layers | Layers selected by energy-guided selection |
| sampling_rate | Channel sampling rate |
| ma | Main Task Accuracy |
| ba | Backdoor Accuracy |
| asr | Attack Success Rate |
| analysis_time | Defense-side analysis time |
| selected_param_ratio | Fraction of analyzed backbone parameters |
| rollback | Whether rollback happened |

---

## 13. Current Status

### Completed

- Implemented/reorganized the project direction.
- Prepared the no-defense baseline setting.
- Prepared SafeSplit-style defense logic.
- Defined the extension idea: Energy-Guided Compact SafeSplit.
- Designed the evaluation metrics for cost-security trade-off.

### In Progress

- Finalizing full SafeSplit-style reproduction.
- Measuring MA and BA under no-defense and defense settings.
- Adding energy-guided layer selection.
- Adding channel sampling.
- Generating cost-security trade-off plots.

### TODO

- [ ] Add a clear `requirements.txt`.
- [ ] Refactor notebook code into Python modules.
- [ ] Add command-line arguments for reproducible experiments.
- [ ] Save all metrics to CSV.
- [ ] Save final plots automatically.
- [ ] Compare full SafeSplit vs compact SafeSplit.
- [ ] Add random-sampling baseline.
- [ ] Add multiple random seeds.
- [ ] Add final report figures.

---

## 14. Limitations

This project is a course reproduction and extension project, not an official implementation of SafeSplit.

Important limitations:

1. The current implementation may simplify some details from the original paper.
2. Some results may differ from the paper because of hardware, random seeds, training length, model split, implementation choices, and dataset preprocessing.
3. The compact extension is an efficiency-oriented heuristic. It does not guarantee that all backdoor signals are always located in the highest-energy layers.
4. The final conclusion should be based on empirical comparison against full SafeSplit and random sampling baselines.

---

## 15. Key Takeaway

The main idea of this project is:

> Full SafeSplit analyzes server-side backbone updates to detect client-side backdoor attacks in Split Learning. This project first reproduces that pipeline, then studies whether a compact subset of layers and channels can preserve similar defense performance with lower defense-side analysis cost.

---

## 16. Reference

```bibtex
@inproceedings{rieger2025safesplit,
  title     = {SafeSplit: A Novel Defense Against Client-Side Backdoor Attacks in Split Learning},
  author    = {Rieger, Phillip and Pegoraro, Alessandro and Kumari, Kavita and Abera, Tigist and Knauer, Jonathan and Sadeghi, Ahmad-Reza},
  booktitle = {Network and Distributed System Security (NDSS) Symposium},
  year      = {2025}
}
```

---

## 17. Disclaimer

This repository is for academic research and coursework only. The backdoor attack implementation is included only to evaluate and understand defense mechanisms in Split Learning.
