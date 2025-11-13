import os
import csv
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace


def get_as_path_iterator(csv_file_path):
    """
    A Python generator that reads a large CSV file row-by-row and yields 
    the content of the 'as_path' column. This is memory-efficient.
    """
    print(f"\nStreaming AS paths from '{csv_file_path}'...")
    line_count = 0
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        # Find the index of the 'as_path' column from the header
        header = next(reader)
        try:
            as_path_index = header.index('as_path')
        except ValueError:
            print("Error: 'as_path' column not found in the CSV header.")
            return # Stop the generator

        # Yield the as_path from each subsequent row
        for row in reader:
            line_count += 1
            if len(row) > as_path_index:
                yield row[as_path_index]
            
            if line_count % 1000000 == 0:
                print(f"  ... streamed {line_count:,} paths")
    print(f"Finished streaming a total of {line_count:,} paths.")


def train_bpe_tokenizer(corpus_file_path, vocab_size, special_tokens, output_path):
    """ Trains a Byte-Pair Encoding (BPE) tokenizer from the CSV corpus. """
    print("\n--- Training BPE Tokenizer ---")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    # Get a fresh iterator for this training session
    path_iterator = get_as_path_iterator(corpus_file_path)

    print(f"Starting BPE training with vocab size {vocab_size}...")
    tokenizer.train_from_iterator(path_iterator, trainer, length=sum(1 for _ in get_as_path_iterator(corpus_file_path)))
    print("Training complete!")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)
    print(f"BPE Tokenizer saved to '{output_path}'")
    print("-" * 30)


def train_wordpiece_tokenizer(corpus_file_path, vocab_size, special_tokens, output_path):
    """ Trains a WordPiece tokenizer from the CSV corpus. """
    print("\n--- Training WordPiece Tokenizer ---")
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    # Get a fresh iterator
    path_iterator = get_as_path_iterator(corpus_file_path)
    
    print(f"Starting WordPiece training with vocab size {vocab_size}...")
    tokenizer.train_from_iterator(path_iterator, trainer, length=sum(1 for _ in get_as_path_iterator(corpus_file_path)))
    print("Training complete!")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)
    print(f"WordPiece Tokenizer saved to '{output_path}'")
    print("-" * 30)


def train_unigram_tokenizer(corpus_file_path, vocab_size, special_tokens, output_path):
    """ Trains a Unigram tokenizer from the CSV corpus. """
    print("\n--- Training Unigram Tokenizer ---")
    tokenizer = Tokenizer(Unigram())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = UnigramTrainer(vocab_size=vocab_size, special_tokens=special_tokens, unk_token="[UNK]")
    
    # Get a fresh iterator
    path_iterator = get_as_path_iterator(corpus_file_path)
    
    print(f"Starting Unigram training with vocab size {vocab_size}...")
    tokenizer.train_from_iterator(path_iterator, trainer, length=sum(1 for _ in get_as_path_iterator(corpus_file_path)))
    print("Training complete!")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)
    print(f"Unigram Tokenizer saved to '{output_path}'")
    print("-" * 30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BGP tokenizers from a processed CSV file.')
    parser.add_argument(
        'csv_file', 
        help='Path to the input CSV file (e.g., data/training_data.csv)'
    )
    parser.add_argument(
        '--vocab_size', 
        type=int, 
        default=30000, 
        help='The desired vocabulary size for the tokenizers.'
    )
    args = parser.parse_args()

    # --- Configuration ---
    SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    
    if not os.path.exists(args.csv_file):
        print(f"Error: Corpus file '{args.csv_file}' not found.")
        print("Please run the data extraction script first.")
    else:
        # --- Run Training ---
        train_bpe_tokenizer(
            corpus_file_path=args.csv_file, 
            vocab_size=args.vocab_size, 
            special_tokens=SPECIAL_TOKENS, 
            output_path="tokenizers/bgp-bpe-tokenizer.json"
        )

        train_wordpiece_tokenizer(
            corpus_file_path=args.csv_file, 
            vocab_size=args.vocab_size, 
            special_tokens=SPECIAL_TOKENS, 
            output_path="tokenizers/bgp-wordpiece-tokenizer.json"
        )
        
        train_unigram_tokenizer(
            corpus_file_path=args.csv_file, 
            vocab_size=args.vocab_size, 
            special_tokens=SPECIAL_TOKENS, 
            output_path="tokenizers/bgp-unigram-tokenizer.json"
        )

        print("\nAll tokenizers trained and saved successfully!")