import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import argparse
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --- 1. Re-use the Dataset and Model Classes ---
# We need to define the exact same classes we used for training
# so that PyTorch knows how to structure the loaded weights.

class ASPathDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=128):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.max_length = max_length
        
        print(f"Loading data from {file_path}...")
        with open(file_path, 'r') as f:
            self.lines = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(self.lines)} lines.")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        # We need both the token IDs for the model and the original text for the label
        text = self.lines[idx]
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids
        
        ids_padded = ids[:self.max_length]
        padding_needed = self.max_length - len(ids_padded)
        ids_padded = ids_padded + [self.pad_token_id] * padding_needed
        
        return torch.tensor(ids_padded), text

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
        # A new, separate method to get just the embedding (the "thought vector")
        with torch.no_grad(): # We don't need to track gradients for inference
            embedded = self.embedding(input_sequence)
            _, (hidden, _) = self.encoder(embedded)
            # Squeeze to remove the 'num_layers' dimension
            return hidden.squeeze(0)

# --- 2. The Main Evaluation Function ---
def main(args):
    # --- Setup ---
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")

    tokenizer_path = f"bgp-{args.tokenizer_type}-tokenizer.json"
    model_path = os.path.join(args.model_dir, f"autoencoder_{args.tokenizer_type}_best.pth")

    if not all(os.path.exists(p) for p in [tokenizer_path, model_path]):
        print(f"Error: Make sure both tokenizer ('{tokenizer_path}') and model ('{model_path}') exist.")
        return

    # --- Load Tokenizer and Data ---
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.token_to_id("[PAD]")
    
    test_dataset = ASPathDataset(tokenizer, "test.txt", max_length=args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # --- Load Trained Model ---
    model = SimpleAutoencoder(vocab_size, padding_idx=pad_token_id).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # CRITICAL: Set model to evaluation mode
    print(f"Model loaded successfully from '{model_path}'")

    # --- 1. Quantitative Evaluation: Calculate Test Loss ---
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    total_test_loss = 0
    with torch.no_grad():
        for batch_ids, _ in test_loader:
            batch_ids = batch_ids.to(device)
            outputs = model(batch_ids)
            loss = criterion(outputs.view(-1, vocab_size), batch_ids.view(-1))
            total_test_loss += loss.item()
    
    avg_test_loss = total_test_loss / len(test_loader)
    print(f"\n--- Quantitative Result ---")
    print(f"Average Test Loss for '{args.tokenizer_type}' tokenizer: {avg_test_loss:.4f}")

    # --- 2. Qualitative Evaluation: Generate Embeddings & Visualize ---
    print("\n--- Generating Embeddings for Visualization ---")
    all_embeddings = []
    all_labels = [] # We will use the Origin AS as the label
    
    # We only need a subset for a clean plot
    num_samples_for_plot = min(args.num_plot_samples, len(test_dataset))
    
    with torch.no_grad():
        for i in range(num_samples_for_plot):
            ids_tensor, text = test_dataset[i]
            ids_tensor = ids_tensor.unsqueeze(0).to(device) # Add batch dimension
            
            embedding = model.encode(ids_tensor)
            all_embeddings.append(embedding.cpu().numpy())
            
            # Extract the Origin AS (the last ASN in the path) as the label
            try:
                origin_as = text.split()[-1]
                all_labels.append(origin_as)
            except IndexError:
                all_labels.append("EMPTY")

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    print(f"Generated {len(all_embeddings)} embeddings. Running t-SNE...")
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # --- Plotting ---
    plt.figure(figsize=(14, 10))
    
    # Create a numeric mapping for labels to colors
    unique_labels = sorted(list(set(all_labels)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = [label_to_id[l] for l in all_labels]
    
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=numeric_labels, cmap='jet', alpha=0.6, s=10)
    
    plt.title(f"t-SNE Visualization of AS_PATH Embeddings (Tokenizer: {args.tokenizer_type.upper()})\nColor-coded by Origin AS", fontsize=16)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    # Create a legend (only show a subset of labels if there are too many)
    if len(unique_labels) < 30:
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                     markerfacecolor=scatter.cmap(scatter.norm(label_to_id[label])))
                          for label in unique_labels]
        plt.legend(handles=legend_handles, title="Origin AS")
        
    plt.grid(True)
    plot_filename = f"tsne_plot_{args.tokenizer_type}.png"
    plt.savefig(plot_filename)
    print(f"Visualization saved to '{plot_filename}'")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained Autoencoder.')
    parser.add_argument('--tokenizer_type', type=str, required=True, choices=['bpe', 'wordpiece', 'unigram'],
                        help='Type of tokenizer used for the model.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation.')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length used during training.')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory where trained models are saved.')
    parser.add_argument('--num_plot_samples', type=int, default=2000, help='Number of samples to use for the t-SNE plot.')
    
    args = parser.parse_args()
    main(args)