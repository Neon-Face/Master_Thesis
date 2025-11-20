
Q1. 
What tokenizer does Word2Vec use? Can you compare BERT, WordPiece, Unigram with it? I think it doesn't use tokenizer to create vocabulary, but just directly embed each word as embedding using one-hot encoding

Q2.
will different tokenizer have impact on performance?

Q3.
lets focus on internet data like IP and ASN, what's the point of tokenize them into subword than word? (what's the subword of an IP or ASN mean?)

Q4.
For this: Your model has learned over months of training that the token for "145.97." is almost always associated with paths ending in AS1103. Suddenly, it sees a path containing tokens for "145.97.34." that ends in AS99999. This is a "grammatical" error. It violates the learned statistical rules. The autoencoder will struggle to reconstruct this path, leading to a **high reconstruction error**, which is your anomaly signal.

Please explain more, i dont understand this. What do you mean by "IP ends in ASN"

Q5.
The prefix is identical, but the origin AS changes depending on your vantage point.

Please tell me the relationship between prefix and AS

Q6.
They found that a BPE-trained vocabulary works surprisingly well with a simple greedy segmentation, performing almost as well as the standard BPE merge-based segmentation.

Whats segmentation? the relationship of tokenizer, BPE and segmentation

Q7.
when you say "Model Size Matters", do you mean the size of the "thought vector"?

Q8.
Finding 2: Vocabulary Construction Method Matters, but There's No Silver Bullet.
They compared different algorithms for building the vocabulary (BPE, WordPiece, Unigram, SaGe, PathPiece).
They found that the top 5-6 tokenizers were all statistically competitive with each other. There was no single algorithm that was demonstrably better than all others (Figure 2).
BPE, the simplest and most common method, performed extremely well and was part of the top competitive group.
Conclusion: The choice of the core algorithm is important, but there is a group of "good enough" methods, and hyper-optimizing between them may yield diminishing returns.

for this, my original plan was to investigate which Vocabulary Construction Method works better for internet data, but now it shows there no bullet point, shall i change my research topic to pre-tokenizer?

Q9.
what does this mean: "A 0x0800 representing the EtherType for IP is semantically different from a 0x0800 that happens to appear in a TCP payload. The tokenizer is blind to this."

Q10
Sub-word of an IPv6 Address might work, but sub-word of an ASN may not because it doesnt have sementic meaning?

Q11
please compare diffusion model and autoregressive model, no analogy

Q12


