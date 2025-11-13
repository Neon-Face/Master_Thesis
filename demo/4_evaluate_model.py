import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import argparse
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# --- 1. Dataset and Model Classes (Must match training script) ---

class ASPathDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=128):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.max_length = max_length
        self.as_paths = []

        print(f"Loading data from {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                as_path_index = header.index('as_path')
                for row in reader:
                    if len(row) > as_path_index and row[as_path_index]:
                        self.as_paths.append(row[as_path_index])
        except FileNotFoundError:
            print(f"Error: Data file not found at {file_path}")
            # Allow creation of an empty dataset
        except ValueError:
             raise ValueError("Error: 'as_path' column not found in CSV header.")

        if self.as_paths:
            print(f"Loaded {len(self.as_paths)} valid AS paths.")
        else:
            print(f"Warning: No AS paths were loaded from {file_path}.")


    def __len__(self):
        return len(self.as_paths)

    def __getitem__(self, idx):
        text = self.as_paths[idx]
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids
        ids = ids[:self.max_length]
        padding_needed = self.max_length - len(ids)
        ids = ids + [self.pad_token_id] * padding_needed
        
        # Return the original text as well for later inspection
        return torch.tensor(ids), text

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

# --- 2. Anomaly Detection Function ---
def detect_anomalies(model, tokenizer, file_path, threshold, device, max_length):
    """
    Reads a file, calculates reconstruction loss for each path, and flags anomalies.
    """
    print(f"\n--- Analyzing for anomalies in '{file_path}' ---")
    print(f"Using anomaly threshold: {threshold:.4f}")

    dataset = ASPathDataset(tokenizer, file_path, max_length=max_length)
    if not dataset.as_paths:
        return

    dataloader = DataLoader(dataset, batch_size=1) # Process one path at a time
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))
    
    anomaly_count = 0
    model.eval()
    with torch.no_grad():
        for i, (batch, text) in enumerate(dataloader):
            batch = batch.to(device)
            outputs = model(batch)
            
            # Reshape for loss calculation, but for a single item batch
            loss = criterion(outputs.view(-1, model.fc_out.out_features), batch.view(-1))
            
            if loss.item() > threshold:
                anomaly_count += 1
                print(f"  -> ANOMALY DETECTED (Loss: {loss.item():.4f}): {text}")
    
    print(f"\nAnalysis complete. Found {anomaly_count} anomalies in {len(dataset)} paths.")

# --- 3. Main Evaluation Function ---
def main(args):
    # --- Setup ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer_path = f"tokenizers/bgp-{args.tokenizer_type}-tokenizer.json"
    model_path = os.path.join(args.model_dir, f"autoencoder_{args.tokenizer_type}_best.pth")

    if not all(os.path.exists(p) for p in [tokenizer_path, model_path, args.val_file]):
        print(f"Error: Make sure tokenizer, model, and validation file exist.")
        return

    # --- Load Tokenizer and Model ---
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.token_to_id("[PAD]")
    
    model = SimpleAutoencoder(vocab_size, padding_idx=pad_token_id).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded successfully from '{model_path}'")

    # --- [Step 1] Calculate Anomaly Threshold from Validation Data ---
    print("\n--- [Step 1/2] Calculating anomaly threshold from normal validation data... ---")
    val_dataset = ASPathDataset(tokenizer, args.val_file, max_length=args.max_length)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    losses = []
    with torch.no_grad():
        for batch, _ in val_loader:
            batch = batch.to(device)
            outputs = model(batch)
            # Calculate loss per item in the batch
            for i in range(batch.shape[0]):
                loss = criterion(outputs[i].unsqueeze(0).view(-1, vocab_size), batch[i].unsqueeze(0).view(-1))
                losses.append(loss.item())

    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    # The threshold is the mean loss plus N standard deviations
    anomaly_threshold = mean_loss + (args.threshold_stdevs * std_loss)
    
    print(f"-> Loss on normal data: Mean={mean_loss:.4f}, StdDev={std_loss:.4f}")
    print(f"-> Anomaly Threshold set at {args.threshold_stdevs} stdevs: {anomaly_threshold:.4f}")

    # --- [Step 2] Run Anomaly Detection ---
    detect_anomalies(model, tokenizer, args.anomaly_file, anomaly_threshold, device, args.max_length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained BGP Autoencoder model.')
    parser.add_argument('--val_file', type=str, required=True, help='Path to the NORMAL validation CSV file to calculate threshold.')
    parser.add_argument('--anomaly_file', type=str, required=True, help='Path to the ANOMALOUS CSV file to test for hijacks.')
    parser.add_argument('--tokenizer_type', type=str, required=True, choices=['bpe', 'wordpiece', 'unigram'], help='Type of tokenizer used for the model.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for evaluation.')
    parser.add_argument('--max_length', type=int, default=64, help='Max sequence length used during training.')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory where trained models are saved.')
    parser.add_argument('--threshold_stdevs', type=float, default=3.0, help='Number of standard deviations above the mean to set the anomaly threshold.')
    
    args = parser.parse_args()
    main(args)