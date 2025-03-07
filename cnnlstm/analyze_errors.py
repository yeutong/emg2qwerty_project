#!/usr/bin/env python
# Error analysis script for EMG-to-keystroke model
# %%
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
import h5py

# Import from emg2qwerty
from emg2qwerty.charset import charset
from emg2qwerty.data import WindowedEMGDataset, EMGSessionData

# Import from project
from model import EMGCNNBiLSTM
from utils import calculate_cer
from config import load_config
from data import create_dataloaders

# %%
def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze model errors on validation dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--window_length', type=int, default=4000, help='Window length used in training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for validation')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--output_dir', type=str, default='error_analysis', help='Directory to save analysis results')
    return parser.parse_args()

def load_model(model_path, device):
    """Load the trained model"""
    char_set = charset()
    model = EMGCNNBiLSTM(
        in_features=2*16*33,  # 2 bands, 16 channels, 33 frequency bins
        num_classes=len(char_set) + 1,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Calculate levenshtein distance matrix and operations
def levenshtein_distance_and_ops(s1, s2):
    """
    Calculate Levenshtein distance and operations (substitution, insertion, deletion)
    between two sequences s1 and s2.
    
    Returns:
    - distance: The Levenshtein distance
    - operations: List of tuples (op, i, j) where op is 'equal', 'replace', 'insert', or 'delete'
                 i is the index in s1, j is the index in s2
    """
    len_s1, len_s2 = len(s1), len(s2)
    
    # Initialize distance matrix
    dp = [[0 for _ in range(len_s2 + 1)] for _ in range(len_s1 + 1)]
    
    # Initialize operations matrix
    ops = [[[] for _ in range(len_s2 + 1)] for _ in range(len_s1 + 1)]
    
    # Base cases
    for i in range(len_s1 + 1):
        dp[i][0] = i
        if i > 0:
            ops[i][0] = ops[i-1][0] + [('delete', i-1, -1)]
    
    for j in range(len_s2 + 1):
        dp[0][j] = j
        if j > 0:
            ops[0][j] = ops[0][j-1] + [('insert', -1, j-1)]
    
    # Fill the matrices
    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
                ops[i][j] = ops[i-1][j-1] + [('equal', i-1, j-1)]
            else:
                deletion = dp[i-1][j] + 1
                insertion = dp[i][j-1] + 1
                substitution = dp[i-1][j-1] + 1
                
                min_op = min(deletion, insertion, substitution)
                dp[i][j] = min_op
                
                if min_op == deletion:
                    ops[i][j] = ops[i-1][j] + [('delete', i-1, -1)]
                elif min_op == insertion:
                    ops[i][j] = ops[i][j-1] + [('insert', -1, j-1)]
                else:  # substitution
                    ops[i][j] = ops[i-1][j-1] + [('replace', i-1, j-1)]
    
    return dp[len_s1][len_s2], ops[len_s1][len_s2]

def calculate_per_char_errors(all_predictions, all_targets):
    """Calculate error metrics for each character using Levenshtein alignment"""
    char_set = charset()
    
    # Initialize counters
    char_counts = defaultdict(int)  # Total occurrences
    char_errors = defaultdict(int)  # Misclassifications
    operation_counts = {'insert': 0, 'delete': 0, 'replace': 0}  # Count by operation type
    confusion_matrix = defaultdict(lambda: defaultdict(int))  # (true, pred) -> count
    
    for pred_seq, target_seq in zip(all_predictions, all_targets):
        # Convert integer labels to characters
        pred_chars = [char_set.label_to_char(idx) if idx < len(char_set.allowed_chars) else "_" 
                     for idx in pred_seq if idx != char_set.null_class]  # Skip blank tokens
        target_chars = [char_set.label_to_char(idx) if idx < len(char_set.allowed_chars) else "_" 
                       for idx in target_seq if idx != char_set.null_class]  # Skip blank tokens
        
        # Calculate Levenshtein distance and operations
        _, operations = levenshtein_distance_and_ops(pred_chars, target_chars)
        
        # Process operations to update statistics
        for op, i, j in operations:
            if op == 'equal':
                # Correct prediction
                true_char = target_chars[j]
                char_counts[true_char] += 1
                confusion_matrix[true_char][true_char] += 1
            elif op == 'replace':
                # Substitution error
                true_char = target_chars[j]
                pred_char = pred_chars[i]
                char_counts[true_char] += 1
                char_errors[true_char] += 1
                confusion_matrix[true_char][pred_char] += 1
                operation_counts['replace'] += 1
            elif op == 'insert':
                # Insertion error (missing character in prediction)
                true_char = target_chars[j]
                char_counts[true_char] += 1
                char_errors[true_char] += 1
                confusion_matrix[true_char]['blank'] += 1
                operation_counts['insert'] += 1
            elif op == 'delete':
                # Deletion error (extra character in prediction)
                pred_char = pred_chars[i]
                # char_counts[pred_char] += 1
                # char_errors[pred_char] += 1
                confusion_matrix['blank'][pred_char] += 1
                operation_counts['delete'] += 1
    
    # Calculate error rates
    char_error_rates = {}
    for char, count in char_counts.items():
        if count > 0:
            char_error_rates[char] = char_errors[char] / count
    
    return char_counts, char_errors, char_error_rates, confusion_matrix, operation_counts

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
    
def plot_confusion_matrix(confusion_matrix, output_dir, max_chars=40):
    """Plot confusion matrix for most common characters"""
    # Get characters with most occurrences
    char_totals = defaultdict(int)
    for true_char, preds in confusion_matrix.items():
        for pred_char, count in preds.items():
            char_totals[true_char] += count
    
    # Select top 20 most common characters
    top_chars = sorted(char_totals.items(), key=lambda x: x[1], reverse=True)[:max_chars]
    top_chars = [c for c, _ in top_chars]
    
    # Create confusion matrix for top chars
    matrix = np.zeros((len(top_chars), len(top_chars)))
    for i, true_char in enumerate(top_chars):
        total = sum(confusion_matrix[true_char].values())
        if total > 0:  # Avoid division by zero
            for j, pred_char in enumerate(top_chars):
                matrix[i, j] = confusion_matrix[true_char][pred_char] / total
    
    # Plot
    plt.figure(figsize=(24, 20))
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

def get_session_timestamps(cfg, data_path, window_idx, window_length=2000, stride=1000, padding=(200, 200)):
    """Get the timestamps for a specific window in the dataset"""
    session_info = cfg['dataset']['val'][0]  # Use first validation session
    session_id = session_info['session']
    file_path = data_path / f"{session_id}.hdf5"
    
    with EMGSessionData(file_path) as session:
        # Calculate the start offset in the original timeseries
        offset = window_idx * stride
        
        # Expand window to include contextual padding
        window_start = max(offset - padding[0], 0)
        window_end = offset + window_length + padding[1]
        window = session[window_start:window_end]
        
        # Extract timestamps for the original (unpadded) window
        timestamps = window[EMGSessionData.TIMESTAMPS]
        start_idx = offset - window_start
        end_idx = (offset + window_length - 1) - window_start
        
        # Get the actual start and end timestamps of the window
        start_t = timestamps[start_idx]
        end_t = timestamps[end_idx]
        
        # Get keystroke data with timestamps
        label_data = session.ground_truth(start_t, end_t)
        
        # Return window timestamps and keystroke data
        return start_t, end_t, label_data.text, label_data.timestamps

def visualize_sequences_with_timestamps(all_predictions, all_targets, input_data, window_indices, cfg, data_path, output_dir, num_samples=5):
    """Visualize prediction sequences with timestamps compared to ground truth"""
    char_set = charset()
    
    # Select random samples to visualize
    if len(all_predictions) > num_samples:
        indices = np.random.choice(len(all_predictions), num_samples, replace=False)
    else:
        indices = range(len(all_predictions))
        
    for i, idx in enumerate(indices):
        pred_seq = all_predictions[idx]
        target_seq = all_targets[idx]
        window_idx = window_indices[idx]
        
        # Convert to characters and remove blank tokens
        pred_chars = [char_set.label_to_char(idx) if idx < len(char_set.allowed_chars) else "_" 
                     for idx in pred_seq if idx != char_set.null_class]
        target_chars = [char_set.label_to_char(idx) if idx < len(char_set.allowed_chars) else "_" 
                       for idx in target_seq if idx != char_set.null_class]
        
        # Get original timestamps
        start_t, end_t, orig_text, keystroke_timestamps = get_session_timestamps(cfg, data_path, window_idx)
        window_duration = end_t - start_t
        
        # Calculate Levenshtein alignment
        _, operations = levenshtein_distance_and_ops(pred_chars, target_chars)
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Create timeline for visualization
        time_axis = np.linspace(start_t, end_t, 1000)  # 1000 points for visualization
        
        # Plot timeline for ground truth
        plt.subplot(3, 1, 1)
        plt.plot(time_axis, np.zeros_like(time_axis), '-', color='gray', alpha=0.3)
        plt.xlim(start_t, end_t)
        plt.ylim(-0.5, 0.5)
        plt.title(f'Timeline (window: {start_t:.2f}s to {end_t:.2f}s, duration: {window_duration:.2f}s)')
        plt.ylabel('Ground Truth')
        
        # Plot ground truth keystrokes at actual timestamps
        y_pos = 0
        for j, char in enumerate(orig_text):
            if j < len(keystroke_timestamps):
                # Plot character at its actual timestamp
                t = keystroke_timestamps[j]
                if start_t <= t <= end_t:  # Only plot if timestamp is within window
                    plt.plot(t, y_pos, 'o', color='blue', markersize=5)
                    plt.text(t, y_pos + 0.1, char, fontsize=10, ha='center')
        
        # Plot predicted keystrokes distributed evenly across the time window
        plt.subplot(3, 1, 2)
        plt.plot(time_axis, np.zeros_like(time_axis), '-', color='gray', alpha=0.3)
        plt.xlim(start_t, end_t)
        plt.ylim(-0.5, 0.5)
        plt.ylabel('Prediction')
        
        # Distribute predictions evenly across the time window
        if len(pred_chars) > 0:
            pred_times = np.linspace(start_t, end_t, len(pred_chars))
            y_pos = 0
            
            # Map the alignment to colors
            char_colors = {}
            for op, i, j in operations:
                if op == 'equal':
                    if i >= 0 and i < len(pred_chars):
                        char_colors[i] = 'green'
                elif op == 'replace':
                    if i >= 0 and i < len(pred_chars):
                        char_colors[i] = 'red'
                elif op == 'insert':
                    if i >= 0 and i < len(pred_chars):
                        char_colors[i] = 'orange'
                elif op == 'delete':
                    pass  # Deletions don't appear in predictions
            
            # Plot each predicted character
            for j, char in enumerate(pred_chars):
                color = char_colors.get(j, 'gray')  # Default to gray if no alignment
                plt.plot(pred_times[j], y_pos, 'o', color=color, markersize=5)
                plt.text(pred_times[j], y_pos + 0.1, char, fontsize=10, ha='center', color=color)
        
        # Plot alignment visualization
        plt.subplot(3, 1, 3)
        plt.axis('off')
        
        # Convert operations to aligned strings
        aligned_pred = []
        aligned_target = []
        
        for op, i, j in operations:
            if op == 'equal':
                if i >= 0 and i < len(pred_chars) and j >= 0 and j < len(target_chars):
                    aligned_pred.append(pred_chars[i])
                    aligned_target.append(target_chars[j])
            elif op == 'replace':
                if i >= 0 and i < len(pred_chars) and j >= 0 and j < len(target_chars):
                    aligned_pred.append(pred_chars[i])
                    aligned_target.append(target_chars[j])
            elif op == 'insert':
                if i >= 0 and i < len(pred_chars):
                    aligned_pred.append(pred_chars[i])
                    aligned_target.append('-')
            elif op == 'delete':
                if j >= 0 and j < len(target_chars):
                    aligned_pred.append('-')
                    aligned_target.append(target_chars[j])
        
        # Display aligned sequences
        plt.text(0.01, 0.8, f'Ground Truth: {"".join(orig_text)}', fontsize=12)
        plt.text(0.01, 0.6, f'Target:       {"".join(target_chars)}', fontsize=12)
        plt.text(0.01, 0.4, f'Aligned Target: {"".join(aligned_target)}', fontsize=12)
        plt.text(0.01, 0.2, f'Aligned Pred:   {"".join(aligned_pred)}', fontsize=12)
        plt.text(0.01, 0.0, f'Prediction:     {"".join(pred_chars)}', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sequence_viz_{i}.png'), dpi=300)
        plt.close()

# %%
def main():
    # For testing in a notebook, manually set the arguments
    # for debugging
    os.chdir('/home/yeutong/CS247/project/cnnlstm/')
    in_notebook = True

    if in_notebook:
        class Args:
            model_path = 'models/cnnbilstm-window-4000--20250305_164506.pth'
            window_length = 4000
            batch_size = 128
            num_workers = 2
            output_dir = 'error_analysis'
    
        args = Args()
    else:
        args = parse_arguments()

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    assert torch.cuda.device_count() == 1, "Only one GPU should be visible"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, device)
    
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
    window_indices = []  # Store window indices for timestamp lookup
    
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(val_loader)):
            # Process all validation data (remove the limit)
            # if batch_idx > 1:
            #     break
            
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
                
                # Approximate window index (this may need adjustment based on your dataset implementation)
                window_indices.append(batch_idx * args.batch_size + i)
    
    # Calculate overall CER
    overall_cer = calculate_cer(all_predictions, all_targets)
    print(f"Overall Character Error Rate: {overall_cer:.4f}")
    
    # Calculate per-character error rates using Levenshtein alignment
    print("Calculating per-character error statistics...")
    char_counts, char_errors, char_error_rates, confusion_matrix, operation_counts = calculate_per_char_errors(
        all_predictions, all_targets
    )
    
    # Print character-specific error rates
    print("\nError rates by character (top 20, min 10 occurrences):")
    error_rates_items = [(c, rate) for c, rate in char_error_rates.items() if char_counts[c] >= 10]
    sorted_items = sorted(error_rates_items, key=lambda x: x[1], reverse=True)
    
    print(f"Number of characters with ≥10 occurrences: {len(sorted_items)}")
    for char, rate in sorted_items[:20]:
        print(f"{char}: {rate:.4f} ({char_errors[char]}/{char_counts[char]})")
    
    # Print operation counts
    print("\nError operations summary:")
    print(f"Substitutions: {operation_counts['replace']}")
    print(f"Insertions: {operation_counts['insert']}")
    print(f"Deletions: {operation_counts['delete']}")
    
    # Plot error rates
    print("\nGenerating visualizations...")
    plot_char_error_rates(char_counts, char_error_rates, args.output_dir)
    
    # Plot confusion matrix
    plot_confusion_matrix(confusion_matrix, args.output_dir)
    
    # Visualize sequences with timestamps
    visualize_sequences_with_timestamps(
        all_predictions, all_targets, all_inputs, 
        window_indices, cfg, data_path, args.output_dir
    )
    
    print(f"Analysis complete. Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 
# %%
