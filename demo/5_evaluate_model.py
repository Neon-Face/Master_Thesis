import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import argparse
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --- 1. Dataset and Model Classes (Required for loading the model) ---
# These class definitions must be identical to the ones used in training
# for PyTorch to correctly map the saved weights to the model architecture.

class ASPathDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=128, with_labels=False):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.max_length = max_length
        self.with_labels = with_labels # Flag to determine what __getitem__ returns
        
        print(f"Loading data from {file_path}...")
        with open(file_path, 'r') as f:
            self.lines = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(self.lines)} lines.")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids
        
        ids_padded = ids[:self.max_length]
        padding_needed = self.max_length - len(ids_padded)
        ids_padded = ids_padded + [self.pad_token_id] * padding_needed
        
        if self.with_labels:
            return torch.tensor(ids_padded), text
        else:
            return torch.tensor(ids_padded)

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
        
    def encode(self, input_sequence):
        # A separate method to get just the embedding (the encoder's output)
        with torch.no_grad():
            embedded = self.embedding(input_sequence)
            _, (hidden, _) = self.encoder(embedded)
            return hidden.squeeze(0)

# --- 2. Helper function to calculate loss ---
def calculate_reconstruction_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            loss = criterion(outputs.view(-1, model.fc_out.out_features), batch.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

# --- 3. The Main Evaluation Function ---
def main(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer_path = f"tokenizer/bgp-{args.tokenizer_type}-tokenizer.json"
    model_path = os.path.join(args.model_dir, f"autoencoder_{args.tokenizer_type}_best.pth")

    if not all(os.path.exists(p) for p in [tokenizer_path, model_path]):
        print(f"Error: Make sure tokenizer ('{tokenizer_path}') and model ('{model_path}') exist.")
        return

    # --- Load Tokenizer ---
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.token_to_id("[PAD]")
    
    # --- Load Trained Model ---
    model = SimpleAutoencoder(vocab_size, padding_idx=pad_token_id).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded successfully from '{model_path}'")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    # --- Evaluation Part 1: Quantitative Loss on Normal Data ---
    print("\n--- [Evaluation 1/3] Calculating baseline loss on normal test data... ---")
    test_dataset = ASPathDataset(tokenizer, "data/test.txt", max_length=args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    normal_loss = calculate_reconstruction_loss(model, test_loader, criterion, device)
    print(f"-> Average Reconstruction Loss on NORMAL data: {normal_loss:.4f}")

    # --- Evaluation Part 2: Anomaly Detection on Hijack Data ---
    print("\n--- [Evaluation 2/3] Calculating loss on anomalous hijack data... ---")
    hijack_file = "data/hijack_event.txt"
    if os.path.exists(hijack_file):
        hijack_dataset = ASPathDataset(tokenizer, hijack_file, max_length=args.max_length)
        hijack_loader = DataLoader(hijack_dataset, batch_size=args.batch_size)
        anomaly_loss = calculate_reconstruction_loss(model, hijack_loader, criterion, device)
        print(f"-> Average Reconstruction Loss on ANOMALOUS data: {anomaly_loss:.4f}")
        
        if anomaly_loss > normal_loss * 1.5: # Example threshold
             print("\nSUCCESS: Anomaly loss is significantly higher than normal loss, as expected.")
        else:
             print("\nWARNING: Anomaly loss is not significantly higher than normal loss.")
    else:
        print(f"'{hijack_file}' not found. Skipping anomaly detection part.")

    # --- Evaluation Part 3: Qualitative Visualization with t-SNE ---
    print("\n--- [Evaluation 3/3] Generating embeddings for t-SNE visualization... ---")
    vis_dataset = ASPathDataset(tokenizer, "data/test.txt", max_length=args.max_length, with_labels=True)
    num_samples_for_plot = min(args.num_plot_samples, len(vis_dataset))
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(num_samples_for_plot):
            ids_tensor, text = vis_dataset[i]
            ids_tensor = ids_tensor.unsqueeze(0).to(device)
            
            embedding = model.encode(ids_tensor)
            all_embeddings.append(embedding.cpu().numpy())
            
            try:
                origin_as = text.split()[-1]
                all_labels.append(origin_as)
            except IndexError:
                all_labels.append("[EMPTY]")

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    print(f"Generated {len(all_embeddings)} embeddings. Running t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, init='pca', learning_rate='auto')
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    plt.figure(figsize=(14, 10))
    unique_labels = sorted(list(set(all_labels)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = [label_to_id[l] for l in all_labels]
    
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=numeric_labels, cmap='jet', alpha=0.6, s=10)
    plt.title(f"t-SNE Visualization of AS_PATH Embeddings (Tokenizer: {args.tokenizer_type.upper()})\nColor-coded by Origin AS", fontsize=16)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    if len(unique_labels) < 25:
        cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)))
        cbar.set_ticklabels(unique_labels)
        cbar.set_label('Origin AS')
        
    plt.grid(True)
    os.makedirs("results", exist_ok=True)
    plot_filename = f"results/tsne_plot_{args.tokenizer_type}.png"
    plt.savefig(plot_filename)
    print(f"Visualization saved to '{plot_filename}'")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained BGP Autoencoder model.')
    parser.add_argument('--tokenizer_type', type=str, required=True, choices=['bpe', 'wordpiece', 'unigram'], help='Type of tokenizer used for the model.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation.')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length used during training.')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory where trained models are saved.')
    parser.add_argument('--num_plot_samples', type=int, default=2000, help='Number of samples to use for the t-SNE plot.')
    
    args = parser.parse_args()
    main(args)