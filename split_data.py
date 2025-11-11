import random
import argparse

def split_corpus(input_file, train_file, val_file, test_file, train_ratio=0.8, val_ratio=0.1):
    """
    Reads a text corpus, shuffles it, and splits it into train, validation, and test sets.

    Args:
        input_file (str): Path to the source corpus file.
        train_file (str): Path to save the training data.
        val_file (str): Path to save the validation data.
        test_file (str): Path to save the test data.
        train_ratio (float): The proportion of data for the training set.
        val_ratio (float): The proportion of data for the validation set.
    """
    print(f"Reading data from '{input_file}'...")
    
    # Read all lines into memory
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    print(f"Read a total of {len(lines)} lines.")

    # --- CRITICAL STEP: Shuffle the data ---
    # This ensures that our train, validation, and test sets are not biased
    # by the order of the data in the original file.
    # We set a random seed to make the shuffle reproducible.
    random.seed(42)
    random.shuffle(lines)
    
    # Calculate split indices
    total_lines = len(lines)
    train_end = int(total_lines * train_ratio)
    val_end = train_end + int(total_lines * val_ratio)

    # Split the data
    train_lines = lines[:train_end]
    val_lines = lines[train_end:val_end]
    test_lines = lines[val_end:]

    # Write the splits to their respective files
    print(f"Writing {len(train_lines)} lines to '{train_file}'...")
    with open(train_file, 'w') as f:
        f.writelines(train_lines)
        
    print(f"Writing {len(val_lines)} lines to '{val_file}'...")
    with open(val_file, 'w') as f:
        f.writelines(val_lines)
        
    print(f"Writing {len(test_lines)} lines to '{test_file}'...")
    with open(test_file, 'w') as f:
        f.writelines(test_lines)
        
    print("\nData splitting complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Splits a text corpus into train, validation, and test sets.'
    )
    parser.add_argument(
        '--corpus_file', 
        default='as_paths_corpus.txt', 
        help='Path to the input corpus file.'
    )
    args = parser.parse_args()

    split_corpus(
        input_file=args.corpus_file,
        train_file='train.txt',
        val_file='validation.txt',
        test_file='test.txt'
    )