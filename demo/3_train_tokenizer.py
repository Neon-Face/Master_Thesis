import os
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_bpe_tokenizer(corpus_files, vocab_size, special_tokens, output_path):
    """
    Trains a Byte-Pair Encoding (BPE) tokenizer from a text corpus.

    Args:
        corpus_files (list): List of paths to the text corpus files.
        vocab_size (int): The desired size of the vocabulary.
        special_tokens (list): A list of special tokens (e.g., [UNK], [CLS]).
        output_path (str): Path to save the trained tokenizer JSON file.
    """
    print("--- Training BPE Tokenizer ---")
    
    # Initialize a BPE tokenizer model
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # The pre_tokenizer splits the input text into words. For AS_PATHs, splitting by whitespace is perfect.
    tokenizer.pre_tokenizer = Whitespace()

    # Initialize a trainer for the BPE model
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    # Train the tokenizer
    print(f"Starting training with vocab size {vocab_size}...")
    tokenizer.train(corpus_files, trainer)
    print("Training complete!")

    # Save the trained tokenizer
    tokenizer.save(output_path)
    print(f"BPE Tokenizer saved to '{output_path}'")
    print("-" * 30)


def train_wordpiece_tokenizer(corpus_files, vocab_size, special_tokens, output_path):
    """
    Trains a WordPiece tokenizer from a text corpus. This is the tokenizer model used by BERT.

    Args:
        corpus_files (list): List of paths to the text corpus files.
        vocab_size (int): The desired size of the vocabulary.
        special_tokens (list): A list of special tokens (e.g., [UNK], [CLS]).
        output_path (str): Path to save the trained tokenizer JSON file.
    """
    print("--- Training WordPiece Tokenizer ---")
    
    # Initialize a WordPiece tokenizer model. It finds sub-word units by maximizing likelihood.
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    
    tokenizer.pre_tokenizer = Whitespace()

    # Initialize a trainer for the WordPiece model
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    # Train the tokenizer
    print(f"Starting training with vocab size {vocab_size}...")
    tokenizer.train(corpus_files, trainer)
    print("Training complete!")

    # Save the trained tokenizer
    tokenizer.save(output_path)
    print(f"WordPiece Tokenizer saved to '{output_path}'")
    print("-" * 30)


def train_unigram_tokenizer(corpus_files, vocab_size, special_tokens, output_path):
    """
    Trains a Unigram tokenizer from a text corpus. This model is often used in models like T5.

    Args:
        corpus_files (list): List of paths to the text corpus files.
        vocab_size (int): The desired size of the vocabulary.
        special_tokens (list): A list of special tokens (e.g., [UNK], [CLS]).
        output_path (str): Path to save the trained tokenizer JSON file.
    """
    print("--- Training Unigram Tokenizer ---")
    
    # Initialize a Unigram tokenizer model. It starts with a large vocabulary and removes tokens.
    # Note: For Unigram, the unk_token is often passed during training.
    tokenizer = Tokenizer(Unigram())

    tokenizer.pre_tokenizer = Whitespace()

    # Initialize a trainer for the Unigram model.
    # The `unk_token` is crucial here.
    trainer = UnigramTrainer(vocab_size=vocab_size, special_tokens=special_tokens, unk_token="[UNK]")
    
    # Train the tokenizer
    print(f"Starting training with vocab size {vocab_size}...")
    tokenizer.train(corpus_files, trainer)
    print("Training complete!")

    # Save the trained tokenizer
    tokenizer.save(output_path)
    print(f"Unigram Tokenizer saved to '{output_path}'")
    print("-" * 30)


if __name__ == '__main__':
    # --- Configuration ---
    # The file(s) that will be used to train the tokenizers.
    CORPUS_FILES = ["data/as_paths_corpus.txt"]
    
    # The desired vocabulary size. A good starting point for this specialized domain.
    VOCAB_SIZE = 30000 
    
    # The standard special tokens required by most Transformer models.
    SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    
    # --- Check if corpus file exists ---
    if not os.path.exists(CORPUS_FILES[0]):
        print(f"Error: Corpus file '{CORPUS_FILES[0]}' not found.")
        print("Please run 'extract_as_paths.py' first to generate the corpus.")
    else:
        # --- Run Training ---
        # Train and save a BPE tokenizer
        train_bpe_tokenizer(
            corpus_files=CORPUS_FILES, 
            vocab_size=VOCAB_SIZE, 
            special_tokens=SPECIAL_TOKENS, 
            output_path="tokenizer/bgp-bpe-tokenizer.json"
        )

        # Train and save a WordPiece tokenizer
        train_wordpiece_tokenizer(
            corpus_files=CORPUS_FILES, 
            vocab_size=VOCAB_SIZE, 
            special_tokens=SPECIAL_TOKENS, 
            output_path="tokenizer/bgp-wordpiece-tokenizer.json"
        )
        
        # Train and save a Unigram tokenizer
        train_unigram_tokenizer(
            corpus_files=CORPUS_FILES, 
            vocab_size=VOCAB_SIZE, 
            special_tokens=SPECIAL_TOKENS, 
            output_path="tokenizer/bgp-unigram-tokenizer.json"
        )

        print("\nAll tokenizers trained and saved successfully!")