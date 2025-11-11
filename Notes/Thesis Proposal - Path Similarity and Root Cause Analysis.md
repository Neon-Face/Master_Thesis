Of course. This is the perfect next step. Presenting a focused, well-researched, and detailed plan is the best way to have a productive discussion with your professor.

Here is a comprehensive article-style proposal for the BGP Hijack Detection project. It incorporates your reasoning, performs a targeted review of recent academic work, clearly defines your novelty, and lays out a concrete project plan.

***

## **Thesis Proposal: Learning the Grammar of Global Routing for Real-Time BGP Hijack Detection**

### **Executive Summary**

The Border Gateway Protocol (BGP) is the foundational pillar of the internet's routing system, yet it remains fundamentally insecure and vulnerable to malicious attacks such as prefix hijacking. While numerous methods for BGP anomaly detection exist, many rely on hand-crafted features that may fail to capture the complex, evolving patterns of global routing. This thesis proposes a novel, end-to-end deep learning methodology to address this challenge. My approach treats the BGP Autonomous System (AS) Path as a sequential language and learns its "grammar" to detect anomalies. The core innovation lies in a two-phase process: **(1)** A data-driven BPE (Byte-Pair Encoding) tokenizer will be trained on a massive corpus of BGP data to learn a vocabulary of meaningful, common AS Path "phrases" or sub-sequences. **(2)** A Transformer-based autoencoder will then be trained on these tokenized paths to learn a deep representation of legitimate routing behavior. Anomalies, such as hijacks, will be detected in real-time as paths that the model fails to reconstruct with high fidelity. I hypothesize this learned-grammar approach will outperform traditional feature-based methods and existing deep learning models by providing a more robust and generalizable understanding of valid routing structure.

---

### **1. Problem Statement & Motivation: The Internet's Trust Problem**

The stability and security of the global internet depend entirely on the BGP protocol. BGP allows independent networks (Autonomous Systems) to announce which IP address prefixes they can reach, forming a chain of trust that creates the global routing table. However, this trust is frequently violated. A BGP hijack occurs when an AS illegitimately announces that it is the origin for an IP prefix it does not own, causing traffic to be misdirected, often for malicious purposes like data theft or denial of service. Famous incidents, such as the 2008 Pakistan-YouTube hijack, have demonstrated the catastrophic potential of these events.

The problem you are solving—BGP anomaly and hijack detection—is a critical, ongoing security issue for every major network on the planet. A successful model from your research wouldn't just be an academic curiosity; it would be the blueprint for a system that an ISP, a cloud provider, or a financial institution would want to deploy immediately. The application is a direct, real-time security control, not just an analysis tool.

### **2. The Data: Unimpeachable Real-World Fidelity**

This is the strongest argument for the BGP use case. The data you will use from **RIPE RIS (Routing Information Service)** and **Route Views** is not a sample or a proxy; it **IS** the real-world data. These public collectors listen to the same BGP messages that route the global internet. When you test your model on a historical dataset of a known hijack, you are working with the literal ground truth of that event. There is zero gap between your experiment and reality in this regard, allowing the results of this thesis to be claimed as directly applicable to real-world scenarios without qualification.

*   **Training Corpus:** A multi-terabyte dataset of BGP UPDATE messages from a "known-good" period (e.g., one month) will be sourced from RIPE RIS archives.
*   **Test Data:** Labeled historical hijack events will be used for evaluation.

### **3. Proposed Methodology: A Two-Phase Deep Learning Approach**

My methodology treats the AS_PATH as a sequence. The core hypothesis is that legitimate routing paths follow a learnable "grammar," while hijacks are "ungrammatical."

#### **Phase 1: The BPE Tokenizer - Learning Routing "Phrases"**
The first and most novel step is to move beyond treating each ASN as an atomic "word." I will train a Byte-Pair Encoding (BPE) tokenizer on the raw AS_PATHs from my training corpus.

*   **Process:** The tokenizer will start with a vocabulary of all individual ASNs. It will then iteratively find the most frequently co-occurring adjacent pair of tokens (e.g., `AS174 AS3356`) and merge them into a new, single token.
*   **Outcome:** The result is a learned vocabulary not just of individual ASNs, but of common, multi-AS "phrases" that represent the superhighways and common peering relationships of the internet. This allows the model to learn that a legitimate pattern like AS Path prepending (`AS701 AS701 AS701`) is a single, normal token, while a malicious loop (`AS-Y -> AS-Z -> AS-Y`) is a sequence of rare, low-probability tokens.

#### **Phase 2: The Transformer Autoencoder - Learning "Grammar"**
The tokenized AS_PATHs will be used to train a Transformer-based autoencoder.

*   **Architecture:** The Transformer is the ideal choice due to its use of **Positional Encodings**, which inherently understands that the position of a token in the path matters (i.e., the origin AS is different from a transit AS).
*   **Training Task:** The model will be trained on a **denoising reconstruction** task. It will be given a corrupted (token-masked) AS_PATH and must learn to reconstruct the original, correct path.
*   **Anomaly Detection:** Once trained, the model will process live BGP updates. A hijack will be flagged if the model exhibits a **high reconstruction error** or assigns a **very low probability** to the observed sequence of tokens, indicating the path is "ungrammatical" and unlike any legitimate path it has ever seen.

### **4. Literature Review and Statement of Novelty**

BGP anomaly detection is a well-established field. My work builds upon prior research while introducing key innovations.

A recent Google Scholar search reveals several key trends:
*   **Feature-Based ML:** Many systems use traditional machine learning (SVM, Random Forests) on hand-crafted features like AS-path length, prefix characteristics, and historical stability (e.g., BGP-iSec).
*   **RNN/LSTM Models:** More recent deep learning approaches have used LSTMs to model the AS_PATH as a sequence (e.g., "DeepBGP"). These models typically treat each ASN as a unique, one-hot encoded token.
*   **Graph Neural Networks (GNNs):** Some state-of-the-art research models the internet as a graph and uses GNNs to detect anomalous links.

My proposal is novel and distinct from these approaches in several critical ways:

| Paper/Approach | Methodology | How My Proposal is Different (Novelty) |
| :--- | :--- | :--- |
| **Traditional Feature-Based ML** | Hand-crafted features (path length, etc.) + SVM/Random Forest. | My approach is **end-to-end**. It *learns* the important features directly from the data, rather than relying on a pre-defined, potentially incomplete set. |
| **LSTM-based Models ("DeepBGP")** | LSTM/RNN on sequences of one-hot encoded ASNs. | **The Tokenizer is my key innovation.** While LSTMs treat the path as a sequence of "words" (ASNs), my BPE tokenizer allows the model to learn the structure of routing "phrases" (multi-AS sub-paths), leading to a richer and more robust understanding of path grammar. |
| **Graph Neural Networks (GNNs)** | Models the internet AS-graph and looks for anomalous edges or properties. | GNNs are excellent at modeling the static graph, but my approach is designed to model the **dynamic, sequential nature of the AS_PATH language itself**. It can capture anomalies that are valid in the graph but highly improbable as a sequence. |

In summary, the primary contribution of this thesis is the introduction of **data-driven, sub-path tokenization** to the problem of BGP security, moving the state-of-the-art from a "word-level" understanding to a more powerful "phrase-level" understanding of routing grammar.

### **5. Project Plan and Timeline**

This project is structured into five distinct phases over an estimated 24-week period.

| Phase | Duration | Key Tasks | Deliverable |
| :--- | :--- | :--- | :--- |
| **1. Foundation & Data Pipeline** | Weeks 1-4 | Finalize literature review. Set up data acquisition pipeline using Py-BGPStream. Download and parse RIPE RIS training/testing data. | A clean, multi-terabyte corpus of AS_PATHs and labeled hijack events. |
| **2. Tokenizer Development** | Weeks 5-8 | Implement and train the BPE tokenizer on the full training corpus. Perform qualitative validation of the learned vocabulary. | A trained, production-ready tokenizer for AS_PATHs. |
| **3. Model Implementation & Training** | Weeks 9-16 | Implement the Transformer-based autoencoder in PyTorch. Train the model on the tokenized training data. Tune hyperparameters. | A trained autoencoder model capable of reconstructing legitimate AS_PATHs. |
| **4. Evaluation & Comparison** | Weeks 17-20 | Implement baseline models (feature-based and LSTM). Evaluate all models against the historical hijack test set. Analyze performance (F1, Precision, Recall). | A comprehensive set of results, graphs, and tables comparing the models. |
| **5. Thesis Writing & Defense** | Weeks 21-24 | Synthesize results, write thesis chapters, and prepare final presentation. | Completed Master's Thesis. |

### **6. Alternative Research Directions**

While the BGP use case is the primary focus due to its high impact and data fidelity, the core methodology of this thesis is highly generalizable. Should the initial research prove challenging, or for future work, the following directions are also viable:

*   **Alternative 1: Traceroute Path Analysis (High Methodological Novelty):** Apply the same tokenizer/autoencoder framework to sequences of IP addresses from RIPE Atlas and CAIDA traceroutes. The goal would be to learn embeddings for network paths to enable novel forms of path similarity analysis and root cause detection for network performance issues. This is a robust academic project with extremely high feasibility.
*   **Alternative 2: DNS Malicious Domain Detection (High Security Impact):** Apply the methodology to character-level sequences of domain names to detect those generated by malware (DGAs). While the real-world impact is enormous, this direction faces significant challenges in acquiring high-fidelity, non-proxy data, making it better suited for future work.

### **7. Conclusion**

This thesis proposes a novel and impactful approach to a critical internet security problem. By combining a data-driven tokenization strategy with a state-of-the-art Transformer architecture, this research aims to significantly advance the field of BGP hijack detection. The project is well-defined, leverages high-fidelity real-world data, and is structured with a clear and achievable plan.

https://ieeexplore.ieee.org/abstract/document/9754706

Questions:
1. teach me why 30.57.1.1 and 130.57.2.2 are topologically "close."
2. whats the point of figuring out **Path Similarity**


```
https://data.ris.ripe.net/rrc03/2025.11/bview.20251111.0000.gz
```