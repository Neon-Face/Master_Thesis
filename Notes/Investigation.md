## Keywords   

Embedding
Internet data 
trace route 
vector db: define a schema, control the scope of input 
different tokenizer 
MRT 
fantail pca 
auto encoder 
how to evaluate tokenizer  


## Basic  

Tokenization breaks text into individual tokens, while Embeddings convert those tokens into numerical vectors that capture semantics and relationships.  

Do we really need tokenizer for internet data like IP address? Why can't we just treat the whole IP address as a single token (Like a word in a sentence), then embed it as a whole. In this case, a trace route record would be a sentence.  
- Too many IP address. (IPv4:  2^32, IPv6: 2^128), the vocabulary size will be too big.
- Hint: character tokenization is often accompanied by a loss of performance. So to get the best of both worlds, transformers models use a hybrid between word-level and character-level tokenization called **subword** tokenization.

What are [embeddings](https://platform.openai.com/docs/guides/embeddings)

Summary of [Tokenizers](https://huggingface.co/docs/transformers/v4.57.1/en/tokenizer_summary#unigram)

[spaCy](https://spacy.io/) and [Moses](http://www.statmt.org/moses/?n=Development.GetStarted) are two popular rule-based tokenizers.
- What do you mean by rule-based? Can I make the rule?

Byte-Pair Encoding (BPE) and Pre-tokenizer: [Here](https://huggingface.co/docs/transformers/v4.57.1/en/tokenizer_summary#byte-pair-encoding-bpe)

Byte-level BPE (Byte Pair Encoding):
idea for IP address, we dont even need 256 bytes, we only need 16 items to be our base vocabulary
- Example for IPv4 address:
	Let's use this small dataset:
	1.  `192.168.1.10`
	2.  `192.168.1.20`
	Start with Bytes
		First, every character is converted to its raw byte value. The common prefix `192.168.1.` becomes a long sequence of bytes:
		`[49, 57, 50, 46, 49, 54, 56, 46, 49, 46, ...]`
	Find and Merge Common Pairs
		The BPE algorithm scans the data and finds the most common adjacent pair of bytes.
		*   **Merge 1:** It sees the pair `[49, 57]` (for "19") is frequent. It merges them into a new single token, let's call it `T1`. Now the sequence starts `[T1, 50, 46, ...]`.
		*   **Merge 2:** It continues, maybe merging `[50, 46]` (for "2.") into a new token `T2`.
		*   **And so on...**
	The Result: Efficient Tokens
		After many merges, entire common chunks of the IPv4 addresses become single tokens.
		*   A new token `T192` might represent the whole sequence for `"192."`.
		*   A token `T168` might represent `"168."`.
		*   A token `T1` might represent `"1."`.
	Ultimately, the long byte sequence for `192.168.1.10` is compressed into a much shorter sequence, like:
	`[T192, T168, T1, 49, 48]` (where `49` and `48` are the original bytes for "10").
	This makes the data much more compact and easier for a model to process.

[WordPiece](https://huggingface.co/docs/transformers/v4.57.1/en/tokenizer_summary#wordpiece): In contrast to BPE, WordPiece does not choose the most frequent symbol pair, but the one that maximizes the likelihood of the training data once added to the vocabulary. It evaluates what it _loses_ by merging two symbols to ensure it’s _worth it_.

Specialized Tokenizer: By seeing 192.168 together thousands of times in network logs, the tokenizer learns that this is a meaningful chunk, a "word" in the language of networking. The model then learns that the token T_192.168 is associated with private, local networks, which is a powerful piece of information.

## Questions

Can I use different tokenizers at the same time? Normally use natural language tokenizer, but when it comes to X-data, use the X-tokenizer for that data. 

Do I really need tokenizer? Since all data are well-structured, why we cannot just directly use feature extraction for each entry. 
- [Anomaly Detection on Devices DNS Queries Using Deep Learning](https://www.sei.cmu.edu/library/anomaly-detection-on-devices-dns-queries-using-deep-learning/)
- [ML for suspicious DNS query detection](https://medium.com/@myth7672/machine-learning-for-suspicious-dns-query-detection-5566f3aa9a52)


![[AI_Scope.jpg]]

Maybe try graph neural network:
- [On Effectiveness of Graph Neural Network Architectures for Network Digital Twins (NDTs)](https://arxiv.org/html/2508.02373v1)

## Traditional Machine Learning (Why is it bad?)
## Deep Learning for Sequences

1. Tokenization:
	1. Create a master list of every unique ASN that has ever appeared in a path.
	2. Each unique ASN is assigned a unique integer ID. (Tokenized)
2. Embedding:
	1. Assign a random vector to each token in your vocabulary.
	2. Learn from context (notices that token 4 and token 3 often appear in similar contexts)
	3. Gradually adjusts the embedding vectors.
	4. Outcome of this step: Each token ID maps to a rich, meaningful vector.
3. Create a Vector for the Entire Path (Sequence Modeling)
	- Simple Way: Pooling / Averaging
		- Pros: Very fast and simple.
		- Cons: It's a "bag of words" approach. It loses the crucial order of the hops. A -> B -> C would have the same final vector as C -> B -> A.
	- Sequence Models (RNN/LSTM/Transformer)
		1. Read the Sequence: The model reads the embedding vectors for your path one by one: vector_hop1, then vector_hop2, etc.
		2. Maintain a "Memory"
		3. Final Representation: The final value of this hidden state, after reading the entire path, becomes the single, context-aware vector for the entire traceroute. This vector now understands that Cogent -> Google is different from Google -> Cogent.
4. Finding Similarity
	1. A new problematic traceroute arrives. It goes through the full pipeline (tokenize -> embed -> sequence model) and is converted into its vector, let's call it V_problem.
	2. Use Cosine Similarity to calculate the "angle" between V_problem and every other vector in your database.
	3. You sort the results and get the top k traceroutes that are most similar to your problem path.

## Graph Neural Networks

1.  Graph Construction
    1.  Create a Node for every unique ASN that has ever appeared in a path.
    2.  Create a directed Edge for every observed connection between two ASNs in a path.
    3.  (Optional) Add weights to each edge to store information like average latency or frequency of the connection.
2.  Initial Node Representation (Initial Features)
    1.  Assign a starting feature vector to every Node (ASN) in the graph.
    2.  This is typically a One-Hot Encoded vector, where each Node's initial feature is simply its own unique identity. This initial vector is called `H^(0)`.
3.  Learning Node Embeddings (Message Passing)
    *   This is the core GNN process. It is performed in layers, where each layer allows nodes to learn from their neighbors.
    *   Single GNN Layer:
        1.  Aggregation ("Gathering Information"): For every node, the GNN collects the feature vectors of all its immediate neighbors and aggregates them into a single summary vector (e.g., by taking the mean).
        2.  Update ("Updating Opinion"): The GNN takes the aggregated neighbor vector and the node's own vector from the previous layer (`H^(0)`) and feeds them through a small neural network. The output becomes the node's new, more informed vector for the next layer (`H^(1)`).
    *   Stacking GNN Layers:
        1.  This process is repeated for multiple layers.
        2.  After 1 layer, a node's vector contains information about its direct neighbors (1-hop neighborhood).
        3.  After 'k' layers, a node's final vector (`H^(k)`) is a rich, dense embedding that represents its role and position within its k-hop neighborhood on the graph.
4.  Create a Vector for the Entire Path
    1.  A new traceroute arrives (e.g., `ASN_X -> ASN_Y -> ASN_Z`).
    2.  Look up the final, fully-trained embedding vectors for each node in the path: `H_X^(k)`, `H_Y^(k)`, `H_Z^(k)`.
    3.  Aggregate these powerful node vectors to get a single vector for the whole path.
        *   Simple Way: Pooling / Averaging
            *   Cons: Loses the specific order of the hops within the path.
        *   Powerful Way: Sequence Models (RNN/LSTM)
            *   Read the sequence of node embedding vectors: `vector_X`, `vector_Y`, `vector_Z`.
            *   The sequence model's final state becomes the single, order-aware vector for the entire traceroute path.
5.  Finding Similarity
    1.  A new problematic traceroute is converted into its final path vector using the full pipeline, let's call it `V_problem`.
    2.  Use Cosine Similarity to calculate the "angle" between `V_problem` and every other path vector in your database.
    3.  You sort the results and get the top k traceroutes that are most structurally similar within the context of the entire network map.
