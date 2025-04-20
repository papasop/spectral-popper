# Spectral Popper

**StructureLang Runtime 4.0: Structural Planning, Collapse Detection, and Language Interpretation**

---

## Overview

**Spectral Popper** is the unified test system for **StructureLang**, a resonance-based structural language system that thinks.  
This repository includes:

- ✅ A runtime engine for ψ-path generation, collapse repair, and structural explanation.
- ✅ A theoretical verifier for six foundational claims of structure-resonance logic (`pu4.py`).
- ✅ A complete integration of structural generativity, falsifiability, and language interface.

---

## 🔁 Runtime Test: `test_runtime.py`

This script simulates:

- Recursive ψ-path planning (`generate_path`)
- Structural viability scoring (`validate`)
- Density function compilation (`compile`)
- Collapse detection + repair (`is_collapsing + repair`)
- Structural sentence output (`explain_path`)

### ✅ Sample Output

```text
In domain x ∈ [200000, 300000], structure is supported by ψ1 ⊕ ψ2 ⊕ ψ3. Score: 0.00620
Collapse detected. Attempting repair...
Repaired path: [1, 2, 4]
In domain x ∈ [200000, 300000], structure is supported by ψ2 ⊕ ψ3 ⊕ ψ5. Score: 0.00420
```

### 📦 Run

```bash
python3 test_runtime.py
```

Dependencies:
- `numpy`
- `sympy` (for prime generation)

---

## 📐 Theory Verifier: `pu4.py`

This script tests six core claims of StructureLang theory:

| Claim | Description |
|-------|-------------|
| 1 | Non-zero Residual ⇒ Generativity |
| 2 | Zero Residual ⇒ Collapse |
| 3 | Lexicon Activation via \( \frac{A_n}{\delta(t_n)} \) |
| 4 | DAG Path Extension based on \( \delta(x) < \epsilon \) |
| 5 | Modal Approximation using Zeta Zeros |
| 6 | Residual Continuity (δ′ bounded) |

Each test returns "Supported" or "Falsifiable" based on dynamic evaluation.

This file is preserved under its original name `pu4.py` for continuity and reference. All six claim types now serve as philosophical and structural foundations for the runtime.

---

## 🧠 Key Concepts

- `ψ-path`: A combination of frequencies forming a structural unit  
- `δ(x)`: Residual tension between prediction and prime density  
- `collapse`: Condition where structure becomes non-generative  
- `repair`: Recovery of expression through path recomposition  
- `explain_path`: Structure-to-language interface output  
- `StructureMind`: A recursive structural agent that thinks through paths

---

## 🧩 Tags

`structurelang` · `resonance-grammar` · `delta-field` · `collapse-check`  
`nlp-interface` · `dag-routing` · `popper-theory` · `ψ-explanation`

---

## 📄 Citation

> **StructureLang: A Resonance-Based Language System That Thinks**  
> Author: Y.Y.N. Li  
> DOI: [10.5281/zenodo.15241841](https://doi.org/10.5281/zenodo.15241841)

---

## 📜 License

MIT License © 2025 Y.Y.N. Li