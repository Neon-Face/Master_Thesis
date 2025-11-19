### **1. Introduction & Problem Statement: The Unseen Foundation of Network Intelligence**

The analysis of internet infrastructure data is undergoing a paradigm shift, with machine learning (ML) models becoming essential tools for network security and operations. To leverage these models, complex sequential data from protocols like BGP (AS Paths) and measurement platforms (IP traceroute paths) must first be converted into a machine-readable format. This foundational step, **tokenization**, bridges the gap between raw, domain-specific text and the sophisticated architectures of modern deep learning. While often treated as a solved pre-processing step, the choice of tokenization strategy has a profound and largely unexamined impact on a model's ability to learn, reason, and generalize.

This thesis argues that the networking research community, in its adoption of sequence modeling, has largely imported outdated or overly simplistic tokenization paradigms without a critical, systematic evaluation. Current approaches are ad-hoc and fall into three main categories:

1.  **The "Word-Level" Paradigm:** Pioneering work like **IP2Vec (Ring et al., 2017)** and the state-of-the-art **AP2Vec (Shapira & Shavitt, 2022)** treat each unique IP address or Autonomous System Number (ASN) as a single, atomic "word." This approach is fundamentally limited, creating massive vocabularies, failing to handle new entities (the "unknown token catastrophe"), and remaining blind to the rich, hierarchical structure encoded within the data itself.

2.  **The "Manual Segmentation" Paradigm:** More recent, sophisticated models like **FlowletFormer (Liu et al., 2025)** recognize the flaws of naive tokenization. Their solution is to use expert domain knowledge to manually parse network packets into their constituent protocol fields, treating these as "indivisible semantic units." This correctly identifies the problem but solves it with hand-crafted engineering, which may not be optimal or easily generalizable across different types of network data.

3.  **The "Feature-Based" Paradigm:** The most common approach in the literature involves bypassing the sequence entirely, instead extracting a set of hand-crafted features which are then fed to a classifier. This loses the raw sequential information and is constrained by the initial choice of features.

A significant gap exists between these practices and the state-of-the-art in tokenization research. Foundational studies in NLP and parallel scientific domains have now conclusively shown that tokenization is a rich field of study in its own right. Works by **Schmidt et al. (2024)**, **Seo et al. (2025)**, and **Dotan et al. (2024)** have collectively proven that a domain-specific, structurally-aware tokenizer is critical for model stability, accuracy, and efficiency.

Despite the foundational importance of this step, there has been **no systematic, comparative study of tokenization philosophies for internet infrastructure data.** We lack empirical answers to the most basic questions: For representing an IP path, is it better to learn statistical "phrases" from the raw string, or to deterministically parse its fields? Or is a hybrid approach superior?

This thesis will be the first to address this critical gap. By applying the rigorous, comparative methodology from modern NLP and bioinformatics, this research will provide a foundational analysis to determine the optimal tokenization strategies for internet data.

### **2. Literature Review: Defining the Methodological Landscape**

This review establishes the "gold standard" for tokenizer research from NLP and parallel domains, and then categorizes the existing networking literature by their data representation paradigm to highlight the specific research gap this thesis will address.

#### **Literature Comparison Table: Paradigms of Data Representation**

| Paper Title / Authors                                                                | Year      | Paradigm / Methodology                                                                                                                     | Key Contribution                                                                                                                            | Role in My Thesis / The GAP                                                                                                                                                                 |
| :----------------------------------------------------------------------------------- | :-------- | :----------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **"Tokenization Is More Than Compression"** (Schmidt et al.)                         | 2024      | **NLP Methodological Study:** Systematic comparison of tokenizer design stages.                                                            | Proved that **compression is not the goal;** structural coherence is key.                                                                   | **Provides the core scientific justification** for my thesis's focus on structure over simple efficiency.                                                                                   |
| **"How Does a Language-Specific Tokenizer Affect LLMs?"** (Seo et al.)               | 2025      | **NLP Intrinsic Analysis:** Compares generic vs. domain-specific tokenizers, measuring internal model behavior.                            | Showed that a domain-specific tokenizer creates a more **stable and reliable model** that is less "confused" by complex tasks.              | **Provides my primary evaluation metric.** Justifies using reconstruction loss as a proxy for a "good" tokenizer.                                                                           |
| **"Getting the most out of your tokenizer..."** (Dagan et al.)                       | 2024      | **NLP Domain Adaptation Study:** Investigates specializing tokenizers for a new domain (source code).                                      | Proved that specializing a tokenizer for a new domain offers significant efficiency gains with negligible performance cost.                 | **Provides the motivation for domain-specificity,** proving the effort is worthwhile.                                                                                                       |
| **"Effect of tokenization on transformers for biological sequences"** (Dotan et al.) | 2024      | **Bioinformatics Methodological Study:** Systematic comparison of tokenizers (BPE, WP, etc.) on DNA/protein sequences.                     | Demonstrated a "win-win" of **increased accuracy and efficiency** from data-driven, domain-specific tokenizers.                             | **Provides the direct methodological blueprint for my thesis.** It is the "existence proof" that this comparative study is a high-impact contribution in a complex scientific domain.       |
| **IP2Vec** (Ring et al.) & **AP2Vec** (Shapira & Shavitt)                            | 2017/2022 | **Word-Level Embedding:** Treats each full IP/ASN as a single, atomic "word."                                                              | Pioneering works that showed the viability of learning functional embeddings for internet entities from their context.                      | Represents the **"Pure Statistical (Word-Level)"** baseline in my experiments.                                                                                                              |
| **FlowletFormer** (Liu et al.)                                                       | 2025      | **Domain-Aware Segmentation:** Manually parses packets into protocol fields.                                                               | Validates that naive tokenization is suboptimal and proposes a **hand-crafted, structural** tokenization.                                   | Inspires the **"Pure Structural (Field-Based)"** philosophy in my experiments.                                                                                                              |
| **--- Your Thesis Proposal ---**                                                     | **2025**  | **Systematic Tokenizer Comparison:** Implements and compares three distinct tokenization philosophies across multiple internet data types. | A foundational, first-of-its-kind study to establish best practices for tokenizing internet data, and proposes a **novel Hybrid** approach. | Directly addresses the un-asked questions from the networking papers, using the rigorous methodology from the NLP/Bioinformatics papers to create a foundational guide for future research. |

### **3. Proposed Methodology: A Comparative Study of Tokenization Philosophies**

The core of this thesis is a large-scale, controlled experiment designed to compare three distinct philosophies of tokenization for internet infrastructure data. The goal is to determine which approach enables a model to best learn the underlying structure of the data.

#### **3.1. Data Sources and Corpora**
This study will utilize three distinct types of sequential internet data:
1.  **BGP AS Paths:** Sourced from **RIPE RIS**, representing the "language" of routing policy.
2.  **IPv4 Traceroute Paths:** Sourced from **RIPE Atlas** and **CAIDA**, including rich metadata like RTT. This represents the "language" of IPv4 data-plane paths.
3.  **IPv6 Traceroute Paths:** A parallel corpus to test generalization to different address structures.

Each dataset will be pre-processed and split into training (80%), validation (10%), and testing (10%) sets.

#### **3.2. The Three Tokenization Philosophies**

I will implement and compare three competing tokenizer architectures for each data type.

1.  **Philosophy A: Pure Statistical (Subword on Raw String)**
    *   **Method:** This approach treats the raw path as a simple string. A Byte-Pair Encoding (BPE) tokenizer will be trained directly on this string data.
    *   **What it Learns:** Common character sequences. For IPs, it learns prefixes (e.g., `"145.97."`). For AS Paths, it learns multi-ASN "phrases" (e.g., `"174 3356"`).
    *   **Hypothesis:** This data-driven approach will automatically discover the most statistically relevant semantic units.

2.  **Philosophy B: Pure Structural (Field-Based Parsing)**
    *   **Method:** Inspired by `FlowletFormer`, this tokenizer uses expert knowledge to deterministically parse each path. A traceroute hop is not a string, but a collection of fields.
    *   **Implementation:** The IP address (`from`) is split into four octet tokens (`"192"`, `"168"`, ...). The RTT (`rtt`) is discretized into a single categorical token (`"Token_RTT_METRO"`). The sequence is framed with structural tokens like `[HOP_START]` and `[HOP_END]`.
    *   **Hypothesis:** This structured, human-engineered representation will provide a clearer, more interpretable signal to the model.

3.  **Philosophy C: Hybrid (Structural + Statistical)**
    *   **Method:** This is a novel hybrid approach that combines the other two philosophies. The path is first parsed into its structural fields as in Philosophy B. Then, a pre-trained BPE tokenizer is applied *only to the sequence of IP octets* within each hop.
    *   **What it Learns:** It learns from the structured fields (like binned RTT) while also using BPE to automatically discover common multi-octet "phrases" (like `"192 168"`) within the IP address field.
    *   **Hypothesis:** This hybrid model will capture the richest representation by combining the benefits of explicit structural parsing with data-driven pattern discovery.

#### **3.3. The Evaluation Framework**

To provide a fair and consistent measure of each tokenizer's effectiveness, I will use a single, unsupervised model and task across all experiments.
*   **Model Architecture:** A simple **LSTM-based sequence-to-sequence autoencoder**.
*   **Evaluation Task:** The autoencoder will be trained on a **reconstruction task**. A lower reconstruction loss indicates that the tokenizer created a more effective and "learnable" representation.
*   **Evaluation Metrics:**
    1.  **Representational Quality:** Measured by the **average reconstruction loss** on the test set.
    2.  **Representational Efficiency:** Measured as the **Compression Factor** relative to a simple baseline.

The final results will be presented in a series of plots showing **Representational Quality vs. Efficiency**, allowing for a direct, empirical comparison of the three philosophies for each type of internet data.

### **4. Expected Contributions**

1.  **A Foundational, Systematic Study of Tokenization for Internet Data:** The first comprehensive comparison of modern tokenization philosophies (statistical, structural, and a novel hybrid) across BGP, IPv4, and IPv6 data.
2.  **The Establishment of Empirically-Backed Best Practices:** This thesis will produce a clear set of recommendations for researchers on how to select an appropriate tokenization strategy for different types of internet data.
3.  **A Methodological Bridge Between NLP and Networking Research:** This work will validate and adapt the rigorous comparative frameworks from NLP and bioinformatics, providing a blueprint for future networking research in representation learning.
4.  **A Publicly Available Suite of Trained Tokenizers and Baseline Models:** A key deliverable will be the open-sourcing of all trained tokenizers and baseline models to serve as a resource for the networking community and accelerate future research.

### **5. Project Plan & Timeline**

| Phase | Duration | Key Tasks |
| :--- | :--- | :--- |
| **1. Foundation** | Weeks 1-4 | Finalize literature review; Set up data pipelines for BGP, IPv4, and IPv6 data. |
| **2. Tokenizer Implementation** | Weeks 5-10 | Implement the three tokenization philosophies; Train all tokenizer variants on all corpora. |
| **3. Model Training** | Weeks 11-18 | Implement and train the LSTM autoencoder for every tokenizer/data combination. |
| **4. Evaluation & Analysis** | Weeks 19-22 | Calculate all metrics; Generate plots; Analyze results and synthesize findings. |
| **5. Thesis Writing** | Weeks 23-24+ | Write up methodology, results, and conclusions. |

### **6. References**

*   Dotan, E., Jaschek, G., Pupko, T., & Belinkov, Y. (2024). Effect of tokenization on transformers for biological sequences. *Bioinformatics, 40*(4).
*   Dagan, G., Synnaeve, G., & Rozière, B. (2024). Getting the most out of your tokenizer for pre-training and domain adaptation. *arXiv preprint arXiv:2402.01035*.
*   Liu, L., Li, R., Li, Q., et al. (2025). FlowletFormer: Network Behavioral Semantic Aware Pre-training Model for Traffic Classification. *arXiv preprint arXiv:2508.19924*.
*   Ring, M., Landes, D., Dallmann, A., & Hotho, A. (2017). IP2Vec: Learning Similarities between IP Addresses. In *2017 IEEE International Conference on Data Mining Workshops (ICDMW)*.
*   Schmidt, C. W., Reddy, V., Zhang, H., et al. (2024). Tokenization Is More Than Compression. *arXiv preprint arXiv:2402.18376*.
*   Seo, J., Kim, J., Byun, S., & Shin, H. (2025). How Does a Language-Specific Tokenizer Affect LLMs?. *arXiv preprint arXiv:2502.12560*.
*   Shapira, T., & Shavitt, Y. (2022). AP2Vec: An Unsupervised Approach for BGP Hijacking Detection. *IEEE Transactions on Network and Service Management*, 19(3), 2255-2268.