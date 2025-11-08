Of course. Here is a formal thesis proposal written in the first person, incorporating all the concepts, arguments, and resources from our discussion. It is structured in a clear, academic format suitable for presentation to your professor.

***

# **Thesis Proposal: Learning Semantic Representations of Network Paths through Data-Driven Tokenization**

## 1. Introduction and Problem Statement

The analysis of network paths, primarily through traceroute data, is fundamental to internet operations, enabling tasks such as performance monitoring, fault diagnosis, and root cause analysis. As the volume of this historical data grows, there is a significant opportunity to apply machine learning techniques to automate these analyses, find meaningful signals in noisy data, and make robust inferences about network behavior.

However, network path data is inherently complex: it is sequential, variable in length, and composed of specialized, structured elements (IP addresses, timeouts). A primary challenge in applying sequence-based machine learning models is determining the optimal method for representing this data. A naive approach, analogous to early methods in Natural Language Processing (NLP), is to treat each unique IP address as a distinct, atomic "word" or token. This method suffers from two critical flaws:

1.  **Massive Vocabulary and Data Sparsity:** The IPv4 address space contains over four billion unique addresses. Even a large dataset will only cover a tiny fraction of this space, leading to an enormous and sparse vocabulary. This makes models inefficient and unable to handle previously unseen IP addresses—a common occurrence in a live network environment.
2.  **Lack of Semantic Representation:** An IP address is not a random identifier; it is a structured entity containing a hierarchy of information (e.g., network prefixes that map to organizations and Autonomous Systems). The naive "one IP, one token" approach discards this rich structural information, treating `192.168.1.1` and `192.168.1.2` as having no more of a relationship than `192.168.1.1` and `8.8.8.8`.

This proposal outlines a research project to systematically address this representational challenge. My work will focus on designing and evaluating tokenization strategies that can learn and exploit the inherent structure of internet data to create powerful, efficient models for network path analysis.

## 2. Literature Review and Foundational Concepts

This research is situated at the intersection of network engineering and modern machine learning, drawing direct inspiration from recent advancements in NLP.

The core methodological precedent for this work comes from the bioinformatics domain. In their paper, **"Effect of tokenization on transformers for biological sequences" (Dotan et al., 2024)**, the authors demonstrate that moving beyond simple character-level tokenization to data-driven methods (like BPE and WordPiece) dramatically improves model performance and efficiency when analyzing DNA and protein sequences. This validates the principle that domain-specific, learned tokenization is a critical component for applying sequence models to non-linguistic data.

Within the networking domain, the **IP2Vec project (Böhm et al., 2021)** established the viability of creating vector embeddings for individual IP addresses. IP2Vec successfully learns the *functional similarity* of IPs based on their communication patterns in NetFlow data. However, its methodology is predicated on a fixed tokenization scheme (one token per IP) and operates on unordered communication data, ignoring the crucial sequential nature of network paths.

My research extends these foundational ideas. While IP2Vec learns the semantics *among* IPs, my work will explore the semantics *inside* the IP address itself. This distinction is fundamental and is motivated by the foundational NLP question of what constitutes a "word" or meaningful token in a given domain, as explored in papers like **"What is a Word? What is a Sentence? Problems of Tokenisation" (Webster & Kit, 1992)**. The goal is to move from a model that only understands the function of an IP to one that also understands its topological location and relationship to its neighbors.

*Additional relevant research includes recent work on representation learning and its application in various domains, as seen in pre-prints such as `https://arxiv.org/abs/2403.06265` and `https://arxiv.org/abs/2402.18376`.*

## 3. Core Thesis Idea and Hypothesis

The central premise of this thesis is that an effective representation of network paths requires capturing semantics at two distinct scales:

1.  **The "Inside IP" Scale (Structural Semantics):** This refers to the hierarchical information encoded within an IP address string, such as the network prefix (`/16`, `/24`), which implies topological proximity and organizational ownership.
2.  **The "Among IPs" Scale (Contextual Semantics):** This refers to the relationships between IPs as they appear sequentially in a traceroute path, revealing the "grammar" of internet routing.

The IP2Vec approach only addresses the contextual scale, and does so without considering sequence. My project will address both simultaneously. It will first solve the "Inside IP" problem through intelligent tokenization and then use a sequence model to learn the contextual patterns.

**Hypothesis:**
> I hypothesize that data-driven tokenization methods (e.g., BPE, WordPiece), which learn a vocabulary of meaningful sub-IP units like network prefixes, will produce superior network path embeddings compared to naive tokenization strategies. These superior embeddings will be demonstrated by:
> 1. Higher accuracy on downstream tasks like path classification.
> 2. Greater model robustness in handling unseen IP addresses.
> 3. Increased computational efficiency by working with a smaller, more meaningful vocabulary.

## 4. Proposed Methodology

To test this hypothesis, I will follow a systematic, multi-stage methodology focused on representation learning using a sequence autoencoder.

#### 4.1. Data Acquisition and Preprocessing
I will source a large-scale traceroute dataset from public repositories such as **RIPE Atlas** and **CAIDA**. This data will be cleaned and normalized into a consistent format, with each traceroute represented as a space-separated sequence of IP hops and timeout markers (`*`).

#### 4.2. The Tokenization Experiment
The core of my experiment will be a comparison of several tokenization strategies:
*   **Baseline 1 ("Full IP"):** The IP2Vec approach, where each unique IP address is a single token.
*   **Baseline 2 ("Octet-level"):** A simple structural tokenizer where each IP is split into its constituent octets and delimiters (e.g., `130`, `.`, `57`, `.`, `22`, `.`, `1`).
*   **Data-Driven Tokenizers:** I will train several tokenizers on the traceroute corpus to learn a domain-specific vocabulary.
    *   **Byte-Pair Encoding (BPE)**
    *   **WordPiece**
    *   **Unigram**
For each data-driven method, I will experiment with multiple vocabulary sizes (e.g., 1k, 5k, 10k) to analyze the trade-off between vocabulary size and performance.

#### 4.3. Model Architecture: Sequence Autoencoder
Instead of a large pre-trained model like BERT, I will use a **Sequence Autoencoder** (implemented with LSTMs or Transformers). The model's purpose is to learn a compressed, fixed-size vector representation (embedding) of a variable-length tokenized traceroute. The autoencoder will consist of an encoder, which maps the input sequence to the embedding, and a decoder, which attempts to reconstruct the original sequence from the embedding.

#### 4.4. Training Task: Denoising Reconstruction
The autoencoder will be trained on a **denoising reconstruction** task. For each input sequence, I will randomly mask or drop a certain percentage of its tokens. The model's objective will be to reconstruct the original, un-corrupted sequence. This forces the model to learn the underlying "grammar" and valid structure of network paths.

#### 4.5. Evaluation Strategy
The quality of the learned embeddings from each tokenizer will be evaluated both quantitatively and qualitatively.
1.  **Downstream Task Performance (Quantitative):** The trained, frozen encoder will be used to generate embeddings for a labeled traceroute dataset (e.g., labeled by destination AS). A simple classifier (e.g., Logistic Regression) will be trained on these embeddings. The primary metric will be the classifier's performance **(Matthew's Correlation Coefficient - MCC)**, which indicates the quality and separability of the embeddings.
2.  **Model Efficiency (Quantitative):** I will measure the **Sequence Length Reduction** achieved by each tokenizer relative to the baseline. Results will be presented on a 2D plot of MCC vs. Length Reduction to visualize the performance-efficiency trade-off.
3.  **Latent Space Visualization (Qualitative):** I will use dimensionality reduction techniques (t-SNE, UMAP) to project the generated path embeddings into a 2D space. Visualizing clear clusters of related paths (e.g., paths going to the same provider) will provide strong intuitive evidence of the model's learning success.
4.  **Vocabulary Inspection (Qualitative):** I will analyze the vocabularies learned by the data-driven tokenizers to confirm that they have discovered meaningful structural units like common network prefixes.

## 5. Expected Outcomes and Contributions

This research is expected to produce the following outcomes:

1.  A systematic comparison of tokenization strategies for network path data.
2.  A trained sequence autoencoder model capable of generating high-quality vector embeddings for traceroutes.
3.  Clear evidence demonstrating the advantages of data-driven, sub-IP tokenization over naive methods.
4.  A novel contribution to the field of network analysis by providing a robust methodology for applying modern sequence modeling techniques to path data, enabling more sophisticated root cause analysis and predictive modeling.

## 6. Tentative Timeline
*   **Weeks 1-4:** Literature Review, Finalizing Proposal, and Data Sourcing.
*   **Weeks 5-8:** Data Cleaning, Preprocessing, and Implementation of Tokenizer Training Pipeline.
*   **Weeks 9-14:** Implementation and Training of Sequence Autoencoder Models for each Tokenizer.
*   **Weeks 15-18:** Implementation and Execution of Downstream Evaluation Tasks and Generation of Results.
*   **Weeks 19-24:** Analysis of Results, Thesis Writing, and Final Presentation.

## 7. References

Böhm, C., Nerling, L., Schuchard, M., & Paxson, V. (2021). **IP2Vec: Learning Similarities Between IP Addresses.** *IEEE European Symposium on Security and Privacy (EuroS&P)*. `https://ieeexplore.ieee.org/abstract/document/11103625`

Dotan, E., Jaschek, G., Pupko, T., & Belinkov, Y. (2024). **Effect of tokenization on transformers for biological sequences.** *Bioinformatics, 40(4)*. `https://arxiv.org/abs/2402.18376`

Webster, J., & Kit, C. (1992). **What is a Word? What is a Sentence? Problems of Tokenisation.** *COLING-92*. `https://www.academia.edu/375399/What_is_a_Word_What_is_a_Sentence_Problems_of_Tokenisation`

*Additional Cited Works:*
`https://arxiv.org/abs/2403.06265`


> 69812379-3024-4761-960a-07bf45249afb

data: https://ieee-dataport.org/documents/tartan-traceroute-dataset#files
