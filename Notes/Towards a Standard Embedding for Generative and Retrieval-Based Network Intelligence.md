### 1. Introduction

The application of Large Language Models (LLMs) to network engineering—"NetAI"—is poised to revolutionize tasks ranging from automated root-cause analysis to generative network simulation. However, a critical foundational gap remains. While LLMs for natural language process text broken into semantic words, internet infrastructure data (BGP AS Paths, IPv6 addresses, Traceroute logs) is treated by current models as either raw strings or arbitrary mathematical sequences.

This lack of a standardized "Internet Token" leads to **Semantic Depletion**. When a model cannot see the structural units of the internet (e.g., Subnets, ASNs), it struggles to generate valid data or retrieve semantically similar network events.

**Thesis Statement:** This research posits that **Semantic-Structural Tokenization**—a hybrid approach that enforces protocol boundaries before applying statistical compression—creates the optimal "Standard Embedding" for internet data. By validating this through a rigorous comparative framework inspired by bioinformatics, this thesis will define the "grammar" of internet infrastructure data.

---

### 2. Literature Review: The State of the Art & The "Semantic Gap"

A review of recent literature reveals a spectrum of approaches, but also a critical flaw: existing methods prioritize either extreme granularity or rigid rules, both of which fail to capture the "Semantic Unit" of the network.

#### Table 1: Summary of Core Literature

| Paper Title                                                                        | Year | Main Methodology                                                             | Key Contribution                                                                                       | The Gap (My Thesis)                                                                                        |
| :--------------------------------------------------------------------------------- | :--- | :--------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------- |
| **Tokenization Is More Than Compression** (Schmidt et al.)                         | 2024 | **PathPiece:** Tests tokenizers by minimizing token count.                   | Proved that high compression $\neq$ better performance. **Pre-tokenization** is the critical step.     |                                                                                                            |
| **How Does a Language-Specific Tokenizer Affect LLMs?** (Seo et al.)               | 2025 | **Extended Tokenizer:** Added Korean vocabulary to Llama-2.                  | Domain-specific tokenization reduces model "confusion" and hallucination confidence.                   | Proves domain adaptation works, but doesn't define *what* the domain unit is for the Internet.             |
| **Getting the most out of your tokenizer...** (Dagan et al.)                       | 2024 | **Tokenizer Transplant:** Swapped tokenizers in pre-trained models.          | Adapting the tokenizer to the specific domain improves efficiency and context window.                  | Established the value of swapping tokenizers, but used standard BPE without field-awareness.               |
| **Effect of tokenization on transformers for biological sequences** (Dotan et al.) | 2024 | **Algorithmic Comparison:** Tested **BPE vs. WordPiece vs. Unigram** on DNA. | **Methodological Blueprint.** Proved that algorithm choice dictates performance in scientific domains. | No one has run this comparison for Internet Data (BGP/Traceroute).                                         |
| **6Diffusion-LM** (Zhao et al.)                                                    | 2024 | **Nybble-Level:** Splits IPv6 into 32 hex digits ($2^{128} \to 32$ tokens).  | Captures perfect structure by treating IPs as math.                                                    | **Semantic Depletion:** Breaks `2001` into `2, 0, 0, 1`. The model loses the concept of the "Prefix."      |
| **6Former** (Liu et al.)                                                           | 2023 | **Byte-Level:** Merges two nybbles into one token (00-FF).                   | Balances sequence length by grouping pairs.                                                            | **Semantic Depletion:** Breaks a 16-bit hextet (`2001`) into `20`, `01`. Fractures subnet semantics.       |
| **FlowletFormer** (Liu et al.)                                                     | 2025 | **Field-Aware:** Splits packets by protocol headers.                         | Respects protocol boundaries.                                                                          | **Rigid Tokenization:** Uses fixed rules inside fields. Misses statistical patterns (like common subnets). |

---

### 3. Problem Statement: Semantic Depletion vs. Rigid Rules

#### A. The Failure of Granularity (6Diffusion & 6Former)
Models like **6Diffusion-LM** and **6Former** achieve high granularity by breaking IPv6 addresses down into Nybbles (4-bit) or Bytes (8-bit).
*   **The Consequence:** This causes **Semantic Depletion**.
*   **Example:** An address contains the prefix `2001`.
    *   *6Diffusion* sees: `[2], [0], [0], [1]`.
    *   *6Former* sees: `[20], [01]`.
*   **Why it fails Generation:** To generate a valid prefix, the model must correctly predict 4 separate tokens in a row. One mistake (`2, 0, 0, 9`) creates an invalid/non-existent network.

#### B. The Failure of Rigidity (FlowletFormer)
**FlowletFormer** attempts to fix this by using **Field Tokenization**. It parses the packet headers (IP, TCP) and treats them as separate segments.
*   **How it works:** It identifies the field, but then applies a **Rigid Rule** to tokenize the content (e.g., "Split IP into 4 bytes").
*   **Example:** A traceroute hop `172.16.254.1` (a private corporate network).
    *   *FlowletFormer* sees: `[AC], [10], [FE], [01]`.
*   **Why it fails Retrieval:** The model sees four disconnected numbers. It does not know that `172.16` (`AC 10`) is a semantic concept ("Private Network"). If we search for "Private Network Outages," this embedding will be far away from `192.168` (another private network) because they share no tokens.

**Thesis Hypothesis:** The optimal solution is **Field-Adaptive BPE**. We must use FlowletFormer's boundary detection, but replace its rigid rules with **BPE**.
*   *My Proposal:* BPE will learn that `172` and `16` appear together frequently.
*   *Result:* `[AC10], [FE], [01]`. The model now has a single token for the subnet.

---

### 4. Methodology

#### Phase 1: Algorithmic Selection (The "Bio-Sequence" Test)
Following the methodology of *Dotan et al. (2024)*, I will first determine which statistical algorithm best compresses internet data without losing information.
*   **Objective:** Determine the baseline algorithm.
*   **Candidates:** **BPE** (Frequency-based), **WordPiece** (Likelihood-based), **Unigram** (Probability-based).
*   **Data:** 10M raw IPv6 addresses and BGP paths.
*   **Outcome:** Selection of the single best algorithm (e.g., BPE) to use in Phase 2.

#### Phase 2: The Granularity Ablation Study
Using the best algorithm from Phase 1, I will train four distinct models to test the "Semantic Depletion" hypothesis.
*   **Model A (Nybble-Level):** The *6Diffusion* baseline. (Vocab size: 16).
*   **Model B (Byte-Level):** The *6Former* baseline. (Vocab size: 256).
*   **Model C (Hextet-Level):** A new baseline forcing 16-bit hextet tokens. (Vocab size: ~65k).
*   **Model D (Field-Adaptive):** My Proposal. Use Regex/Parsing to isolate fields, then run BPE *inside* the field.

#### Phase 3: Evaluation (Utility & Standardization)
To validate that Model D creates the optimal "Standard Embedding," I will evaluate the models on three tasks mirroring real-world NetAI applications.

1.  **Generative Realism (The "Hallucination" Test)**
    *   **Task:** Prompt the model with a traceroute start node and generate the next 5 hops.
    *   **Metric:**
        *   **Syntactic Validity:** % of generated IPs that are valid strings.
        *   **Topological Consistency:** Do generated hops belong to adjacent ASNs in the real internet topology? (Checked against CAIDA databases).
    *   *Hypothesis:* Model D will hallucinate less because it treats subnets as atomic units.

2.  **Downstream Classification (The "Outage" Test)**
    *   **Task:** Detect network outages from RIPE Atlas traceroute logs.
    *   **Method:** Train a linear classifier on top of the frozen embeddings.
    *   **Metric:** F1-Score and **Few-Shot Efficiency** (performance with only 10/50/100 examples).
    *   *Hypothesis:* Model D will learn faster because outage patterns (like `* * *`) are distinct tokens, not fractured bytes.

3.  **Semantic Search & Retrieval (The "Standard Embedding" Test)**
    *   **Task:** Vector Search.
    *   **Query:** A traceroute showing a routing loop in a specific ISP (e.g., Comcast).
    *   **Retrieval:** Does the system return *other* loops in Comcast (Semantic Match) or just random paths that share similar digits (Textual Match)?
    *   **Metric:** Mean Reciprocal Rank (MRR) and t-SNE Visualization of anomalies.