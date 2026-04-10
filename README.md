# non-transformer-llm
Building LLM-like intelligence without transformers using concept graphs, multi-hop reasoning, and lightweight neural networks.
---------------------------------------------------------------------------------------------------------------------------------
# 🧠 GF-SDM — Alternative AI Architecture (No Transformers)

GF-SDM is an experimental AI system that explores **language generation and reasoning without transformers**.

Instead of relying on large-scale neural architectures, this project builds intelligence through a **hybrid approach**:

* Symbolic reasoning (concept graphs)
* Multi-hop inference
* Self-expanding knowledge
* Lightweight neural networks (NumPy-based)

---

## 🚀 Vision

To explore whether **LLM-like behavior** can emerge from:

* Structured knowledge
* Concept relationships
* Small, efficient neural systems

Without requiring:

* GPUs
* Huge datasets
* Transformer architectures

---

## 🧩 Architecture Evolution

### 🔹 v1–v4 — Pattern Learning

* N-gram based language generation
* Basic dataset ingestion
* Early generative behavior

---

### 🔹 v5–v7 — Factual + Symbolic AI

* Q&A-based factual system
* Self-check loop for correctness
* Concept extraction and linking

---

### 🔹 v8 — Self-Building Brain

* Concept normalization (synonyms → canonical form)
* Graph-based knowledge representation
* Multi-hop reasoning (A → B → C → D)
* Automatic dataset expansion from raw text
* Persistent memory system

👉 Key idea:

> Intelligence emerges from **connections between concepts**

---

### 🔹 v9 — Neural Brain (NumPy)

* Fully custom neural network (no frameworks)
* Embedding → Hidden → Softmax architecture
* Trained on factual sequences
* Generates language word-by-word (LLM-style)

👉 Key idea:

> Weights act as a **compact knowledge representation**

---
## 🔹 v10 — Hybrid Brain

Combines symbolic reasoning and neural generation.

Pipeline:
question → reasoning engine → concept chain → neural generator → answer

Key improvements:
- Multi-word context (3 words)
- Better answer accuracy
- Reduced random generation errors
- 
- v11 — Controlled Hybrid
Anchored generation
Strict fallback validation
Q→A training

## 🧠 Core Concepts

### 1. Symbolic Reasoning

* Knowledge stored as structured facts
* Concepts linked through relationships
* Transparent and explainable reasoning

---

### 2. Multi-hop Thinking

Example:

```
gravity → mass → matter → energy → light
```

The system builds reasoning chains across concepts.

---

### 3. Concept Normalization

Different words mapped to same idea:

```
massive → large
produce → generate
```

Improves reasoning accuracy.

---

### 4. Neural Generation (v9)

* Learns word transitions from data
* Generates answers dynamically
* No transformer, no external libraries

---

## ⚡ Features

* ✅ Pure Python (NumPy only for v9)
* ✅ Runs on low-end systems (CPU, 8GB RAM)
* ✅ No external AI frameworks
* ✅ Explainable reasoning (v8)
* ✅ Generative language (v9)
* ✅ Self-expanding knowledge base

---

## 🛠️ Usage

### Run Symbolic Brain (v8)

```bash
python3 gf_sdm_v8.py
```

### Run Neural Brain (v9)

```bash
python3 gf_sdm_v9.py
```

---

### Train Neural Brain

```bash
/train 50
```

### Train from File

```bash
/train_file book.txt
```

---

### Multi-hop Reasoning

```bash
/hop gravity light
```

---

### Inspect Knowledge

```bash
/inspect gravity
```

---

## 📊 Example

**Input:**

```
what connects neurons and memory
```

**Output:**

```
Reasoning chain (memory → synaptic → neurons):
Synaptic plasticity enables memory formation.
Neurons form connections that store memories.
```

---

## 🔥 Why Not Transformers?

This project explores an alternative path:

| Transformers        | GF-SDM                |
| ------------------- | --------------------- |
| Massive scale       | Lightweight           |
| Black-box reasoning | Explainable           |
| GPU required        | CPU-friendly          |
| Implicit knowledge  | Explicit + structured |

---

## 🎯 Goals

* Build efficient AI systems
* Explore non-transformer architectures
* Combine symbolic + neural intelligence
* Enable learning on low-resource devices

---

## 📌 Future Work

* Hybrid system (v8 reasoning + v9 generation)
* Better semantic understanding
* Self-learning from mistakes
* Larger datasets (10k–100k facts)
* Improved reasoning scoring

---

## 🤝 Contribution

This is an experimental research project.
Ideas, feedback, and improvements are welcome.

---

## 👤 Author

**Arjun R**

Independent AI exploration focused on:

* Alternative architectures
* Lightweight intelligence systems
* Human-like reasoning models

---

## 🌌 Final Thought

> Intelligence may not require massive scale —
> it may emerge from **structure, memory, and connection**.
