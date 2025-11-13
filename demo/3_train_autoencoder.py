import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import argparse
import os
import time
import csv

# --- 1. The Dataset Class ---
class ASPathDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=128):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.max_length = max_length
        self.as_paths = []

        print(f"Loading and preparing data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            try:
                as_path_index = header.index('as_path')
            except ValueError:
                raise ValueError("Error: 'as_path' column not found in CSV header.")

            for row in reader:
                if len(row) > as_path_index and row[as_path_index]:
                    self.as_paths.append(row[as_path_index])

        print(f"Loaded {len(self.as_paths)} valid AS paths.")

    def __len__(self):
        return len(self.as_paths)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode(self.as_paths[idx])
        ids = encoding.ids
        ids = ids[:self.max_length]
        padding_needed = self.max_length - len(ids)
        ids = ids + [self.pad_token_id] * padding_needed
        return torch.tensor(ids)

# --- 2. The Autoencoder Model Architecture ---
class SimpleAutoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, padding_idx=0):
        super(SimpleAutoencoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_sequence):
        embedded = self.embedding(input_sequence)
        _, (hidden_state, cell_state) = self.encoder(embedded)
        decoder_output, _ = self.decoder(embedded, (hidden_state, cell_state))
        predictions = self.fc_out(decoder_output)
        return predictions

# --- 3. The Main Training and Evaluation Function ---
def main(args):
    # --- Setup ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # MPS doesn't support pin_memory, so we handle it.
        pin_memory = False 
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        pin_memory = True
    else:
        device = torch.device("cpu")
        pin_memory = True
    print(f"Using device: {device}")

    # --- Tokenizer Loading ---
    tokenizer_filename = f"tokenizers/bgp-{args.tokenizer_type}-tokenizer.json"
    if not os.path.exists(tokenizer_filename):
        print(f"Error: Tokenizer file not found at '{tokenizer_filename}'")
        return
        
    tokenizer = Tokenizer.from_file(tokenizer_filename)
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.token_to_id("[PAD]")
    print(f"Tokenizer '{args.tokenizer_type}' loaded. Vocab size: {vocab_size}")

    # --- Data Loading ---
    train_dataset = ASPathDataset(tokenizer, args.train_file, max_length=args.max_length)
    val_dataset = ASPathDataset(tokenizer, args.val_file, max_length=args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=pin_memory)

    # --- Model Initialization ---
    model = SimpleAutoencoder(vocab_size, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, padding_idx=pad_token_id).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    best_val_loss = float('inf')
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        total_train_loss = 0
        
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs.view(-1, vocab_size), batch.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()

            if (i + 1) % 1000 == 0:
                print(f"  Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                
                # --- THIS IS THE FIX ---
                # The typo `--1` has been corrected to `-1`
                loss = criterion(outputs.view(-1, vocab_size), batch.view(-1))
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        epoch_duration = time.time() - start_time
        
        print(f"\nEpoch {epoch+1}/{args.epochs} | Time: {epoch_duration:.2f}s")
        print(f"  -> Train Loss: {avg_train_loss:.4f}")
        print(f"  -> Val Loss:   {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(output_dir, f"autoencoder_{args.tokenizer_type}_best.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Validation loss improved. Saving model to '{model_save_path}'")
            
    print("\n--- Training Complete ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an Autoencoder on AS_PATHs.')
    parser.add_argument('--train_file', type=str, required=True, help='Path to the training CSV file.')
    parser.add_argument('--val_file', type=str, required=True, help='Path to the validation CSV file.')
    parser.add_argument('--tokenizer_type', type=str, required=True, choices=['bpe', 'wordpiece', 'unigram'],
                        help='Type of trained tokenizer to use.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length for padding/truncation.')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of the token embeddings.')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of the LSTM hidden state.')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save trained models.')
    
    args = parser.parse_args()
    main(args)