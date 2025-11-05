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

what is BGP routing table,  BGP message , 

```
   https://data.ris.ripe.net/rrc03/2025.10/updates.20251014.0000.gz
```