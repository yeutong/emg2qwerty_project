#!/usr/bin/env python
# Evaluation script for EMG-to-keystroke model
# %%
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import os

# Import from emg2qwerty
from emg2qwerty.charset import charset

# Import from project
from model import EMGCNNBiLSTM
from utils import calculate_cer
from config import load_config
from data import create_dataloaders

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--window_length', type=int, default=4000, help='Window length used in training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=2, help='GPU ID to use')
    return parser.parse_args()

def load_model(model_path, num_input_features, device):
    """Load the trained model"""
    model = EMGCNNBiLSTM(
        in_features=num_input_features,
        num_classes=len(charset()) + 1,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, data_loader, device, dataset_name="validation"):
    """Evaluate the model on the given dataset"""
    print(f"\nEvaluating on {dataset_name} dataset...")
    
    # For storing predictions and targets
    all_predictions = []
    all_targets = []
    
    # For detailed error analysis
    char_correct = 0
    char_total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=f'Evaluating {dataset_name}')
        for batch_idx, batch in enumerate(progress_bar):
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            target_lengths = batch['target_lengths'].to(device)
            
            # Forward pass
            emissions = model(inputs, input_lengths)
            
            # Apply log_softmax to get log probabilities
            log_probs = torch.nn.functional.log_softmax(emissions, dim=2)
            
            # Calculate emission lengths
            T_diff = inputs.shape[0] - emissions.shape[0]
            emission_lengths = input_lengths - T_diff
            
            # Get predictions and targets
            predictions = log_probs.argmax(dim=2).cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            # Store predictions and targets for CER calculation
            for i in range(inputs.size(1)):  # Loop through batch
                pred_length = emission_lengths[i].item()
                pred_seq = predictions[:pred_length, i]
                target_length = target_lengths[i].item()
                target_seq = targets_np[:target_length, i]
                
                all_predictions.append(pred_seq)
                all_targets.append(target_seq)
                
                # Count characters for accuracy calculation
                min_len = min(pred_length, target_length)
                for j in range(min_len):
                    char_total += 1
                    if pred_seq[j] == target_seq[j]:
                        char_correct += 1
    
    # Calculate CER using the provided utility function
    cer = calculate_cer(all_predictions, all_targets)
    
    # Calculate simple character accuracy
    char_accuracy = char_correct / char_total if char_total > 0 else 0
    
    # Print results
    print(f"{dataset_name} CER: {cer:.4f}")
    print(f"{dataset_name} Character Accuracy: {char_accuracy:.4f}")
    
    return cer, char_accuracy
# %%
def main():
    args = parse_arguments()
    
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(f"Using device: {device}")
    
    # Calculate input features based on channel drop rate
    num_hands = 2
    num_channels = 16
    num_features = 33
    num_input_features = num_hands * num_channels * num_features
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, num_input_features, device)
    
    # Load data
    emg2qwerty_path = Path(__file__).parent.parent / "emg2qwerty"
    config_path = emg2qwerty_path / "config/user/single_user.yaml"
    cfg = load_config(config_path)
    data_path = emg2qwerty_path / "data"
    
    # Create dataloaders
    print("Loading datasets...")
    _, val_loader, test_loader = create_dataloaders(
        cfg, 
        data_path, 
        128, 
        num_workers=2, 
        use_keystroke_augmentation=False, 
    )
    # model = load_model('models/' + 'cnnbilstm-window-4000-baseline-20250309_123659.pth', num_input_features, 0.4, device)
    
    # Evaluate on validation dataset
    val_cer, val_accuracy = evaluate_model(model, val_loader, device, "validation")
    
    # Evaluate on test dataset
    test_cer, test_accuracy = evaluate_model(model, test_loader, device, "test")
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Validation CER: {val_cer:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test CER: {test_cer:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # # Save results to a file
    # results_dir = Path("evaluation_results")
    # results_dir.mkdir(exist_ok=True)
    
    # model_name = Path(args.model_path).stem
    # with open(results_dir / f"{model_name}_results.txt", "w") as f:
    #     f.write(f"Model: {args.model_path}\n")
    #     f.write(f"Window Length: {args.window_length}\n")
    #     f.write(f"Dropout: {args.dropout}\n")
    #     f.write(f"Downsample Rate: {args.downsample_rate}\n")
    #     f.write(f"Channel Drop Rate: {args.channel_drop_rate}\n")
    #     f.write(f"Number of Sessions: {args.num_sessions}\n\n")
    #     f.write(f"Validation CER: {val_cer:.4f}\n")
    #     f.write(f"Validation Character Accuracy: {val_accuracy:.4f}\n\n")
    #     f.write(f"Test CER: {test_cer:.4f}\n")
    #     f.write(f"Test Character Accuracy: {test_accuracy:.4f}\n")
    
    # print(f"Results saved to {results_dir / f'{model_name}_results.txt'}")

if __name__ == "__main__":
    main()
