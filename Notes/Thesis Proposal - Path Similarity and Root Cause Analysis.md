
>I want to investigate IP level data, instead of ASNs. So original thought is embed each unique IP address. for ipv4 there are 2^32, for ipv6 there are 2^128, it is impossible to put each of them as a node. so, now i need to design a tokenization method to tokenize IP address, instead of storing each possible ip address as a node, i decide to store ip-address-fragment as a node (for example 142.251 shows up a lot in my dataset, so it become a token), and the GNN is trained for learning the meaning among each ip-address-fragment i have. And when it comes to traceroute similarity look up. it rebuild the similar route based on ip-address-fragment.


>historical data, noisy data, want to make inferences, find correct signal, build up to the proposal
>Control variables, tokenizer papers, 
> Effective of tokenization on transformers for bio sequences
> Build the testbed simplest tokenizer, model, benchmarks
> make PLAN, focus on 80% first
> ip- prefix - asn - org

>https://ieeexplore.ieee.org/abstract/document/11103625
>https://www.academia.edu/375399/What_is_a_Word_What_is_a_Sentence_Problems_of_Tokenisation
>https://arxiv.org/abs/2403.06265
>https://arxiv.org/abs/2402.18376
>
## AutoEncoder:

[Intro](https://medium.com/@piyushkashyap045/a-comprehensive-guide-to-autoencoders-8b18b58c2ea6)

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

## BPE doesn't work for data X?

Limit the input data to only internet data we want to explore (only IP address in this case)