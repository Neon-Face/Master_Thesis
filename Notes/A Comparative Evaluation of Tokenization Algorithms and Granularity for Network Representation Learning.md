### 1. Literature Review & The "Semantic Gap"

We have access to a wealth of raw internet data (BGP, IPv6, Traffic), but current Machine Learning models struggle to "read" it effectively. A review of recent literature reveals a spectrum of approaches, but also a critical flaw: **Semantic Depletion**.

#### Table 1: Summary of Core Literature

| Paper Title                                                                        | Year | Main Methodology / Paradigm                                                                        | Key Limitation / Gap                                                                                                                                                                |
| :--------------------------------------------------------------------------------- | :--- | :------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tokenization Is More Than Compression** (Schmidt et al.)                         | 2024 | **PathPiece:** Tests tokenizers by mathematically minimizing token count.                          | Proved that high compression $\neq$ better performance. Pre-tokenization (splitting text intelligently) is the most critical step.==(Need to find the semantic unit)==              |
| **How Does a Language-Specific Tokenizer Affect LLMs?** (Seo et al.)               | 2025 | **Extended Tokenizer:** Adds Korean vocabulary to Llama-2.                                         | Showed that domain-specific tokenization reduces model "confusion" (entropy) and hallucination, even if accuracy metrics are similar.==(Showing it is worth to upgrade tokenizer)== |
| **Getting the most out of your tokenizer...** (Dagan et al.)                       | 2024 | **Tokenizer Transplant:** Swaps a general tokenizer for a code-specific one in pre-trained models. | Proved that adapting the tokenizer to the specific domain (Code) drastically ==improves efficiency and context window.==                                                            |
| **Effect of tokenization on transformers for biological sequences** (Dotan et al.) | 2024 | **Algorithmic Comparison:** Systematically tests **BPE vs. WordPiece vs. Unigram** on DNA/Protein. | **This is the methodological blueprint for my thesis.** ==It proves that the choice of algorithm dictates model performance in scientific domains.==                                |
| **6Diffusion-LM** (Zhao et al.)                                                    | 2024 | **Nybble-Level:** Splits IPv6 into 32 separate hex digits ($2^{128} \to 32$ tokens).               | **Semantic Depletion:** By treating `2001` as `2`, `0`, `0`, `1`, it loses the concept of the prefix. ==The model sees digit, not network structure.==                              |
| **6Former** (Liu et al.)                                                           | 2023 | **Byte-Level:** Merges two nybbles into one token (00-FF).                                         | **Semantic Depletion:** It breaks the standard 16-bit hextet (`2001`) into `20`, `01`. ==The semantic meaning of the subnet is fractured.==                                         |
| **FlowletFormer** (Liu et al.)                                                     | 2025 | **Field-Aware:** Splits packets by protocol headers, then tokenizes content.                       | **The closest to a solution.** It respects field boundaries but relies on rigid rules rather than learning statistical subwords.                                                    |

---

### 2. Problem Statement: The "Semantic Depletion" of IPv6

Current state-of-the-art models like **6Diffusion-LM** and **6Former** suffer from **Semantic Depletion**. They achieve high granularity by breaking IPv6 addresses down into Nybbles (4-bit) or Bytes (8-bit).

*   **The Problem:** Network administrators and routing protocols operate on **Hextets** (16-bit segments, e.g., `2001`, `db8`, `aaaa`).
*   **==Example:==** An address contains the prefix `2001`.
    *   *6Diffusion* sees: `[2, 0, 0, 1]` (4 tokens). The model must learn that these 4 distinct tokens relate to each other.
    *   *6Former* sees: `[20, 01]` (2 tokens). The model must learn that `20` and `01` form a prefix.
*   **The Consequence:** The model wastes capacity relearning basic syntax that should be inherent in the token. It is "reading letters" instead of "reading words."

**FlowletFormer** attempts to fix this. It treats the Protocol Header as the boundary.
*   **How FlowletFormer works:** It parses the packet. If it sees an IP Header, it extracts the raw hex. If the field is long (like a payload), it force-splits it into 4-digit hex tokens.
*   *Example:* Input `0x45000034`. FlowletFormer tokenizes this as `[4, 5, 00, 0034]`.
*   **The Limitation:** While FlowletFormer respects boundaries, it creates a fixed vocabulary. It does not use statistical learning (like BPE) to find the *most common* subnets or patterns automatically.

---

### 3. Methodology

My research will systematically determine the optimal **Semantic Unit** for internet data. I will follow the rigorous experimental design of *Dotan et al. (2024)*, adapted for Network Engineering.

#### Phase 1: Algorithmic Comparison (The "Bio-Sequence" Approach)
Just as *Dotan et al.* tested which algorithm was best for DNA, I will test which statistical algorithm is best for the "Language of the Internet" (BGP paths and IPv6 addresses).

*   **Goal:** Determine which algorithm minimizes entropy while maximizing semantic retention.
*   **The Candidates:**
    1.  **BPE (Byte-Pair Encoding):** Merges the most frequent adjacent pairs. (e.g., might learn `2001` is a token).
    2.  **WordPiece:** Merges pairs that maximize the likelihood of the training data.
    3.  **Unigram:** Starts with a massive vocabulary and trims down the least useful tokens.
*   **Experiment:** Train these three tokenizers on a massive corpus of IPv6 addresses and BGP AS Paths. Measure which one produces the most efficient and "readable" vocabulary.

#### Phase 2: Defining the Semantic Unit (Granularity)
Once the best algorithm (e.g., BPE) is selected, I will test the **Granularity Level**. I will train four distinct models, each representing data differently, to see which one learns best.

*   **Model A (Nybble-Level):** The *6Diffusion* baseline. (Vocab size: 16).
*   **Model B (Byte-Level):** The *6Former* baseline. (Vocab size: 256).
*   **Model C (Hextet-Level):** My hypothesis. We force the tokenizer to respect the 16-bit hextet (`xxxx:xxxx`). (Vocab size: ~65k).
*   **Model D (Field-Adaptive):** The *FlowletFormer* evolution. We use Regex to separate protocol fields (ASN, Prefix, Community), and then run BPE *only inside those fields*.

#### Phase 3: Evaluation (Reconstruction & Understanding)
How do we know which is best? We don't just look at compression; we look at **understanding**.

1.  **Reconstruction Loss (Perplexity):** Train a lightweight Transformer (BERT-style) on a Masked Language Modeling (MLM) task. Mask 15% of the IP address or AS Path.
    *   *Metric:* Which tokenizer allows the model to predict the missing piece with the lowest error?
2.  **Semantic Clustering:** Extract the embeddings (vectors).
    *   *Visual Test:* Do IPv6 addresses from the same subnet cluster together? Do ASNs from the same country cluster together?
    *   *Hypothesis:* The "Hextet" or "Field-Adaptive" models will show tighter semantic clusters than the "Nybble" models.

---

### 4. Expected Contribution

1.  **The "Internet Token" Standard:** This thesis will provide the first empirical evidence defining whether the atomic unit of the internet is the *Nybble*, the *Byte*, or the *Field*, solving the debate between *6Diffusion* and *FlowletFormer*.
2.  **Open Source Tokenizer Suite:** A release of pre-trained BPE/Unigram tokenizers specifically optimized for BGP and IPv6, allowing future researchers to skip the "Tokenizer Transplant" phase.
3.  **Methodological Roadmap:** Establishing a scientific standard for how to preprocess network data, moving the field away from ad-hoc heuristics toward linguistically valid representation learning.