#!/usr/bin/env python
# Error analysis script for EMG-to-keystroke model

import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
import seaborn as sns
from tqdm import tqdm

# Import from emg2qwerty
from emg2qwerty.charset import charset
from emg2qwerty.data import WindowedEMGDataset

# Import from project
from model import EMGCNNBiLSTM
from utils import calculate_cer
from config import load_config
from data import create_dataloaders

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze model errors on validation dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--window_length', type=int, default=4000, help='Window length used in training')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate used in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for validation')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--output_dir', type=str, default='error_analysis', help='Directory to save analysis results')
    return parser.parse_args()

def load_model(model_path, dropout, device):
    """Load the trained model"""
    char_set = charset()
    model = EMGCNNBiLSTM(
        in_features=2*16*33,  # 2 bands, 16 channels, 33 frequency bins
        num_classes=len(char_set) + 1,
        dropout=dropout
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def calculate_per_char_errors(all_predictions, all_targets):
    """Calculate error metrics for each character"""
    char_set = charset()
    
    # Initialize counters
    char_counts = defaultdict(int)  # Total occurrences
    char_errors = defaultdict(int)  # Misclassifications
    confusion_matrix = defaultdict(lambda: defaultdict(int))  # (true, pred) -> count
    
    for pred_seq, target_seq in zip(all_predictions, all_targets):
        # Align sequences (simplified approach - assuming similar lengths)
        min_len = min(len(pred_seq), len(target_seq))
        
        for i in range(min_len):
            true_idx = target_seq[i]
            pred_idx = pred_seq[i]
            
            # Skip null class
            if true_idx == char_set.null_class:
                continue
                
            # Get character representations
            true_char = char_set.label_to_char(true_idx) if true_idx < len(char_set.allowed_chars) else "UNK"
            pred_char = char_set.label_to_char(pred_idx) if pred_idx < len(char_set.allowed_chars) else "UNK"
            
            # Update counts
            char_counts[true_char] += 1
            if true_idx != pred_idx:
                char_errors[true_char] += 1
                
            # Update confusion matrix
            confusion_matrix[true_char][pred_char] += 1
            
    # Calculate error rates
    char_error_rates = {}
    for char, count in char_counts.items():
        if count > 0:
            char_error_rates[char] = char_errors[char] / count
        
    return char_counts, char_errors, char_error_rates, confusion_matrix

def plot_char_error_rates(char_counts, char_error_rates, output_dir):
    """Plot error rates for each character"""
    # Filter characters with at least 10 occurrences for meaningful statistics
    filtered_chars = [c for c, count in char_counts.items() if count >= 10]
    filtered_rates = [char_error_rates[c] for c in filtered_chars]
    
    # Sort by error rate
    sorted_indices = np.argsort(filtered_rates)[::-1]  # Descending order
    sorted_chars = [filtered_chars[i] for i in sorted_indices]
    sorted_rates = [filtered_rates[i] for i in sorted_indices]
    sorted_counts = [char_counts[c] for c in sorted_chars]
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot error rates
    bars = plt.bar(range(len(sorted_chars)), sorted_rates, color='skyblue')
    
    # Add count labels above bars
    for i, (bar, count) in enumerate(zip(bars, sorted_counts)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f'n={count}',
            ha='center', va='bottom',
            rotation=45, fontsize=8
        )
    
    plt.xlabel('Character')
    plt.ylabel('Error Rate')
    plt.title('Error Rate by Character (for chars with ≥10 occurrences)')
    plt.xticks(range(len(sorted_chars)), sorted_chars, rotation=45)
    plt.ylim(0, min(1.0, max(sorted_rates) * 1.2))  # Cap at 1.0
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'char_error_rates.png'), dpi=300)
    plt.close()
    
def plot_confusion_matrix(confusion_matrix, output_dir):
    """Plot confusion matrix for most common characters"""
    # Get characters with most occurrences
    char_totals = defaultdict(int)
    for true_char, preds in confusion_matrix.items():
        for pred_char, count in preds.items():
            char_totals[true_char] += count
    
    # Select top 20 most common characters
    top_chars = sorted(char_totals.items(), key=lambda x: x[1], reverse=True)[:20]
    top_chars = [c for c, _ in top_chars]
    
    # Create confusion matrix for top chars
    matrix = np.zeros((len(top_chars), len(top_chars)))
    for i, true_char in enumerate(top_chars):
        total = sum(confusion_matrix[true_char].values())
        if total > 0:  # Avoid division by zero
            for j, pred_char in enumerate(top_chars):
                matrix[i, j] = confusion_matrix[true_char][pred_char] / total
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix, 
        annot=True, 
        fmt='.2f', 
        xticklabels=top_chars, 
        yticklabels=top_chars,
        cmap='Blues'
    )
    plt.xlabel('Predicted Character')
    plt.ylabel('True Character')
    plt.title('Confusion Matrix (Normalized by Row)')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
def visualize_sequences(all_predictions, all_targets, input_data, output_dir, num_samples=5):
    """Visualize some prediction sequences compared to ground truth"""
    char_set = charset()
    
    # Select random samples to visualize
    if len(all_predictions) > num_samples:
        indices = np.random.choice(len(all_predictions), num_samples, replace=False)
    else:
        indices = range(len(all_predictions))
        
    for i, idx in enumerate(indices):
        pred_seq = all_predictions[idx]
        target_seq = all_targets[idx]
        
        # Convert to characters
        pred_chars = [char_set.label_to_char(idx) if idx < len(char_set.allowed_chars) else "_" 
                     for idx in pred_seq]
        target_chars = [char_set.label_to_char(idx) if idx < len(char_set.allowed_chars) else "_" 
                       for idx in target_seq]
        
        # Create figure
        plt.figure(figsize=(15, 6))
        
        # Plot sequences
        time_points_pred = range(len(pred_chars))
        time_points_target = range(len(target_chars))
        
        plt.subplot(2, 1, 1)
        
        # Plot target sequence
        for j, char in enumerate(target_chars):
            plt.text(j, 0.5, char, fontsize=12, ha='center')
        plt.xlim(-1, len(target_chars))
        plt.ylim(0, 1)
        plt.title(f'Ground Truth: {"".join(target_chars)}')
        plt.axis('off')
        
        plt.subplot(2, 1, 2)
        
        # Plot prediction sequence
        for j, char in enumerate(pred_chars):
            color = 'green' if j < len(target_chars) and pred_chars[j] == target_chars[j] else 'red'
            plt.text(j, 0.5, char, fontsize=12, ha='center', color=color)
        plt.xlim(-1, len(pred_chars))
        plt.ylim(0, 1)
        plt.title(f'Prediction: {"".join(pred_chars)}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sequence_viz_{i}.png'), dpi=300)
        plt.close()

def main():
    args = parse_arguments()

    # Set device
    device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, args.dropout, device)
    
    # Load config and data
    emg2qwerty_path = Path(__file__).parent.parent / "emg2qwerty"
    config_path = emg2qwerty_path / "config/user/single_user.yaml"
    cfg = load_config(config_path)
    data_path = emg2qwerty_path / "data"
    
    # Load only validation data
    _, val_loader, _ = create_dataloaders(
        cfg, data_path, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        use_keystroke_augmentation=False
    )
    
    print("Running inference on validation data...")
    
    # Run inference
    all_predictions = []
    all_targets = []
    all_inputs = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            inputs = batch['inputs'].to(device)  # (T, N, bands, channels, freq)
            targets = batch['targets'].to(device)  # (T, N)
            input_lengths = batch['input_lengths'].to(device)
            target_lengths = batch['target_lengths'].to(device)
            
            # Forward pass
            emissions = model(inputs, input_lengths)
            
            # Calculate emission lengths
            T_diff = inputs.shape[0] - emissions.shape[0]
            emission_lengths = input_lengths - T_diff
            
            # Simple greedy decoding
            predictions = emissions.argmax(dim=2).cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            # Store predictions and targets for analysis
            for i in range(inputs.size(1)):  # Loop through batch
                pred_length = emission_lengths[i].item()
                pred_seq = predictions[:pred_length, i]
                target_len = target_lengths[i].item()
                target_seq = targets_np[:target_len, i]
                
                all_predictions.append(pred_seq)
                all_targets.append(target_seq)
                all_inputs.append(inputs[:, i].cpu())
    
    # Calculate overall CER
    overall_cer = calculate_cer(all_predictions, all_targets)
    print(f"Overall Character Error Rate: {overall_cer:.4f}")
    
    # Calculate per-character error rates
    print("Calculating per-character error statistics...")
    char_counts, char_errors, char_error_rates, confusion_matrix = calculate_per_char_errors(
        all_predictions, all_targets
    )
    
    # Print character-specific error rates
    print("\nError rates by character (top 20, min 10 occurrences):")
    error_rates_items = [(c, rate) for c, rate in char_error_rates.items() if char_counts[c] >= 10]
    sorted_items = sorted(error_rates_items, key=lambda x: x[1], reverse=True)
    
    for char, rate in sorted_items[:20]:
        print(f"{char}: {rate:.4f} ({char_errors[char]}/{char_counts[char]})")
    
    # Plot error rates
    print("\nGenerating visualizations...")
    plot_char_error_rates(char_counts, char_error_rates, args.output_dir)
    
    # Plot confusion matrix
    plot_confusion_matrix(confusion_matrix, args.output_dir)
    
    # Visualize sequences
    visualize_sequences(all_predictions, all_targets, all_inputs, args.output_dir)
    
    print(f"Analysis complete. Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 