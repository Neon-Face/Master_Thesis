#demo/4_train_autoencoder.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import argparse
import os
import time

# --- 1. The Dataset Class ---
class ASPathDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=128):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.max_length = max_length
        
        print(f"Loading and preparing data from {file_path}...")
        with open(file_path, 'r') as f:
            self.lines = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(self.lines)} lines.")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode(self.lines[idx])
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
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=1)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=1)
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
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    tokenizer_filename = f"tokenizer/bgp-{args.tokenizer_type}-tokenizer.json"
    if not os.path.exists(tokenizer_filename):
        print(f"Error: Tokenizer file not found at '{tokenizer_filename}'")
        return
        
    tokenizer = Tokenizer.from_file(tokenizer_filename)
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.token_to_id("[PAD]")
    print(f"Tokenizer '{args.tokenizer_type}' loaded. Vocab size: {vocab_size}")

    # --- Data Loading ---
    train_dataset = ASPathDataset(tokenizer, "data/train.txt", max_length=args.max_length)
    val_dataset = ASPathDataset(tokenizer, "data/validation.txt", max_length=args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    # --- Model Initialization ---
    model = SimpleAutoencoder(vocab_size, padding_idx=pad_token_id).to(device)
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
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            
            # --- CORRECTED LINE ---
            # The target tensor `batch` must be flattened to match the output shape.
            loss = criterion(outputs.view(-1, vocab_size), batch.view(-1))
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                
                # --- CORRECTED LINE ---
                # The same correction is needed here.
                loss = criterion(outputs.view(-1, vocab_size), batch.view(-1))
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        epoch_duration = time.time() - start_time
        
        print(f"\nEpoch {epoch+1}/{args.epochs} | Time: {epoch_duration:.2f}s")
        print(f"  -> Train Loss: {avg_train_loss:.4f}")
        print(f"  -> Val Loss:   {avg_val_loss:.4f}")

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(output_dir, f"autoencoder_{args.tokenizer_type}_best.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Validation loss improved. Saving model to '{model_save_path}'")
            
    print("\n--- Training Complete ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an Autoencoder on AS_PATHs.')
    parser.add_argument('--tokenizer_type', type=str, required=True, choices=['bpe', 'wordpiece', 'unigram'],
                        help='Type of tokenizer to use.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length for padding/truncation.')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save trained models.')
    
    args = parser.parse_args()
    main(args)