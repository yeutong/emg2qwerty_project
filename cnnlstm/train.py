# %%
# %load_ext autoreload 
# %autoreload 2 
# %%
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision.transforms import Compose
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime  # Import datetime

# Import from emg2qwerty
from emg2qwerty.charset import charset
from emg2qwerty.data import WindowedEMGDataset
from emg2qwerty.transforms import ToTensor, LogSpectrogram, SpecAugment, RandomBandRotation

# Import from project
from model import EMGCNNBiLSTM, TDSConvCTC
from utils import calculate_cer
from config import load_config
from data import create_dataloaders

# Set visible GPU to GPU 3
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#assert torch.cuda.device_count() == 1, "Only one GPU should be visible"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %%
batch_size = 128
num_workers = 0

emg2qwerty_path = Path(__file__).parent.parent / "emg2qwerty"

# Load config
config_path = emg2qwerty_path / "config/user/single_user.yaml"
cfg = load_config(config_path)

# Load data
data_path = emg2qwerty_path / "data"
train_loader, val_loader, test_loader = create_dataloaders(cfg, data_path, batch_size, num_workers)


# %%
def train_model(model, train_loader, val_loader, device, epochs=40, lr=0.001, weight_decay=1e-4):
    """
    Train the TDSConvCTC model
    """
    model = model.to(device)

    # CTC loss for sequence prediction - use the same blank token as they do
    criterion = nn.CTCLoss(blank=charset().null_class, reduction='mean', zero_infinity=True)

    # Adam optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler matching their configuration
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.003,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1  # 10% warmup
    )
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        # Store predictions and targets for the last 10 batches
        last_predictions = []
        last_targets = []

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_idx, batch in enumerate(progress_bar):
            inputs = batch['inputs'].to(device)  # (T, N, bands, channels, freq)
            targets = batch['targets'].to(device)  # (T, N)
            input_lengths = batch['input_lengths'].to(device)
            target_lengths = batch['target_lengths'].to(device)

            # Forward pass
            emissions = model(inputs, input_lengths)

            # Calculate emission lengths (account for temporal difference due to conv layers)
            T_diff = inputs.shape[0] - emissions.shape[0]
            emission_lengths = input_lengths - T_diff

            # Compute loss
            loss = criterion(
                log_probs=emissions,
                targets=targets.transpose(0, 1),  # (T, N) -> (N, T) as they do
                input_lengths=emission_lengths,
                target_lengths=target_lengths
            )

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            scheduler.step()

            # Update statistics
            batch_loss = loss.item()
            train_loss += batch_loss
            train_batches += 1

            # Log training loss to WandB
            wandb.log({"train_loss": batch_loss})

            # Store predictions and targets for the last 10 batches
            if batch_idx >= len(train_loader) - 10:  # Only keep the last 10 batches
                predictions = emissions.argmax(dim=2).cpu().numpy()  # Simple greedy decoding
                targets_np = targets.cpu().numpy()

                for i in range(inputs.size(1)):  # Loop through batch
                    pred_length = emission_lengths[i].item()
                    pred_seq = predictions[:pred_length, i]
                    target_seq = targets_np[:target_lengths[i].item(), i]

                    last_predictions.append(pred_seq)
                    last_targets.append(target_seq)

            # Update progress bar
            progress_bar.set_postfix({'loss': batch_loss})

        avg_train_loss = train_loss / train_batches

        # Calculate and log training CER for the last 10 batches
        if len(last_predictions) > 0:
            train_cer = calculate_cer(last_predictions, last_targets)
            wandb.log({"train_cer": train_cer})

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for batch_idx, batch in enumerate(progress_bar):
                inputs = batch['inputs'].to(device)
                targets = batch['targets'].to(device)
                input_lengths = batch['input_lengths'].to(device)
                target_lengths = batch['target_lengths'].to(device)

                # Forward pass
                emissions = model(inputs, input_lengths)

                # Calculate emission lengths
                T_diff = inputs.shape[0] - emissions.shape[0]
                emission_lengths = input_lengths - T_diff

                # Compute loss
                loss = criterion(
                    log_probs=emissions,
                    targets=targets.transpose(0, 1),
                    input_lengths=emission_lengths,
                    target_lengths=target_lengths
                )

                # Update statistics
                batch_loss = loss.item()
                val_loss += batch_loss
                val_batches += 1

                # For CER calculation
                predictions = emissions.argmax(dim=2).cpu().numpy()  # Simple greedy decoding
                targets_np = targets.cpu().numpy()

                # Store predictions and targets for CER calculation
                for i in range(inputs.size(1)):  # Loop through batch
                    pred_length = emission_lengths[i].item()
                    pred_seq = predictions[:pred_length, i]
                    target_seq = targets_np[:target_lengths[i].item(), i]

                    all_predictions.append(pred_seq)
                    all_targets.append(target_seq)
                
                
                # Log validation loss and CER to WandB
                wandb.log({"val_loss": batch_loss})

                # Update progress bar
                progress_bar.set_postfix({'loss': batch_loss})


        # Calculate epoch metrics
        avg_val_loss = val_loss / val_batches
        val_cer = calculate_cer(all_predictions, all_targets)
        wandb.log({"val_cer": val_cer})

        # Print epoch summary
        print(f'Epoch {epoch+1}/{epochs} Summary:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train CER: {train_cer:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val CER: {val_cer:.4f}')

    return {
        'final_train_loss': avg_train_loss,
        'final_val_loss': avg_val_loss,
        'final_train_cer': train_cer,
        'final_val_cer': val_cer
    }



def train_model_mixup(model, train_loader, val_loader, device, epochs=40, lr=0.001, weight_decay=1e-4, mixup_alpha=0.2):
    """
    Train the model with MixUp augmentation
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (cuda/cpu)
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        mixup_alpha: Alpha parameter for MixUp (0 = no mixup)
    """
    model = model.to(device)

    # CTC loss for sequence prediction - use the same blank token as they do
    criterion = nn.CTCLoss(blank=charset().null_class, reduction='mean', zero_infinity=True)

    # Adam optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler matching their configuration
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.003,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1  # 10% warmup
    )
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        # Store predictions and targets for the last 10 batches
        last_predictions = []
        last_targets = []

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_idx, batch in enumerate(progress_bar):
            inputs = batch['inputs'].to(device)  # (T, N, bands, channels, freq)
            targets = batch['targets'].to(device)  # (T, N)
            input_lengths = batch['input_lengths'].to(device)
            target_lengths = batch['target_lengths'].to(device)

            # Apply MixUp augmentation if alpha > 0
            if mixup_alpha > 0:
                # Sample lambda from Beta(alpha, alpha)
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                
                # Create random permutation for mixing
                # We need to shuffle along the batch dimension (dim=1)
                batch_size = inputs.size(1)
                index = torch.randperm(batch_size, device=device)
                
                # Mix the inputs - preserving the time-first dimension structure
                # inputs shape is (T, N, bands, channels, freq)
                mixed_inputs = lam * inputs + (1 - lam) * inputs[:, index]
                
                # Forward pass with mixed inputs
                emissions = model(mixed_inputs, input_lengths)
                
                # Calculate emission lengths (account for temporal difference due to conv layers)
                T_diff = inputs.shape[0] - emissions.shape[0]
                emission_lengths = input_lengths - T_diff
                
                # For CTC loss, we need to compute two separate losses and mix them
                # First loss with original targets
                loss1 = criterion(
                    log_probs=emissions,
                    targets=targets.transpose(0, 1),
                    input_lengths=emission_lengths,
                    target_lengths=target_lengths
                )
                
                # Second loss with mixed targets
                loss2 = criterion(
                    log_probs=emissions,
                    targets=targets[:, index].transpose(0, 1),
                    input_lengths=emission_lengths,
                    target_lengths=target_lengths[index]
                )
                
                # Mix the losses
                loss = lam * loss1 + (1 - lam) * loss2
            else:
                # Forward pass without MixUp
                emissions = model(inputs, input_lengths)
                
                # Calculate emission lengths
                T_diff = inputs.shape[0] - emissions.shape[0]
                emission_lengths = input_lengths - T_diff
                
                # Compute loss
                loss = criterion(
                    log_probs=emissions,
                    targets=targets.transpose(0, 1),
                    input_lengths=emission_lengths,
                    target_lengths=target_lengths
                )

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            scheduler.step()

            # Update statistics
            batch_loss = loss.item()
            train_loss += batch_loss
            train_batches += 1

            # Log training loss to WandB
            wandb.log({"train_loss": batch_loss})

            # Store predictions and targets for the last 10 batches
            if batch_idx >= len(train_loader) - 10:  # Only keep the last 10 batches
                # For CER calculation, use unmixed inputs for more accurate evaluation
                with torch.no_grad():
                    if mixup_alpha > 0:
                        # Recompute predictions with original inputs for CER calculation
                        pure_emissions = model(inputs, input_lengths)
                        predictions = pure_emissions.argmax(dim=2).cpu().numpy()
                    else:
                        predictions = emissions.argmax(dim=2).cpu().numpy()  # Simple greedy decoding
                
                targets_np = targets.cpu().numpy()

                for i in range(inputs.size(1)):  # Loop through batch
                    pred_length = emission_lengths[i].item()
                    pred_seq = predictions[:pred_length, i]
                    target_seq = targets_np[:target_lengths[i].item(), i]

                    last_predictions.append(pred_seq)
                    last_targets.append(target_seq)

            # Update progress bar
            progress_bar.set_postfix({'loss': batch_loss})

        avg_train_loss = train_loss / train_batches

        # Calculate and log training CER for the last 10 batches
        if len(last_predictions) > 0:
            train_cer = calculate_cer(last_predictions, last_targets)
            wandb.log({"train_cer": train_cer})

        # Validation phase - no changes needed here as we don't apply MixUp during validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for batch_idx, batch in enumerate(progress_bar):
                inputs = batch['inputs'].to(device)
                targets = batch['targets'].to(device)
                input_lengths = batch['input_lengths'].to(device)
                target_lengths = batch['target_lengths'].to(device)

                # Forward pass
                emissions = model(inputs, input_lengths)

                # Calculate emission lengths
                T_diff = inputs.shape[0] - emissions.shape[0]
                emission_lengths = input_lengths - T_diff

                # Compute loss
                loss = criterion(
                    log_probs=emissions,
                    targets=targets.transpose(0, 1),
                    input_lengths=emission_lengths,
                    target_lengths=target_lengths
                )

                # Update statistics
                batch_loss = loss.item()
                val_loss += batch_loss
                val_batches += 1

                # For CER calculation
                predictions = emissions.argmax(dim=2).cpu().numpy()  # Simple greedy decoding
                targets_np = targets.cpu().numpy()

                # Store predictions and targets for CER calculation
                for i in range(inputs.size(1)):  # Loop through batch
                    pred_length = emission_lengths[i].item()
                    pred_seq = predictions[:pred_length, i]
                    target_seq = targets_np[:target_lengths[i].item(), i]

                    all_predictions.append(pred_seq)
                    all_targets.append(target_seq)
                
                # Log validation loss and CER to WandB
                wandb.log({"val_loss": batch_loss})

                # Update progress bar
                progress_bar.set_postfix({'loss': batch_loss})

        # Calculate epoch metrics
        avg_val_loss = val_loss / val_batches
        val_cer = calculate_cer(all_predictions, all_targets)
        wandb.log({"val_cer": val_cer})

        # Print epoch summary
        print(f'Epoch {epoch+1}/{epochs} Summary:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train CER: {train_cer:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val CER: {val_cer:.4f}')

    return {
        'final_train_loss': avg_train_loss,
        'final_val_loss': avg_val_loss,
        'final_train_cer': train_cer,
        'final_val_cer': val_cer
    }



# %%


# Initialize model with parameters matching their configuration
# model = TDSConvCTC(
#     in_features=528,  # 33 freq bins * 16 channels
#     mlp_features=[384],
#     block_channels=[24, 24, 24, 24],
#     kernel_width=32
# )
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--note', type=str, default='')
parser.add_argument('--window_length', type=int, default=4000)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=1e-3)
args = parser.parse_args()

model = EMGCNNBiLSTM(
    in_features=2*16*33, 
    num_classes=len(charset()) + 1,
    dropout=0.4
)

# Initialize WandB
wandb.init(
    project="emg2qwerty", 
    entity="vijayasree-ucla",
    name=f'cnnbilstm-window-{args.window_length}-{args.note}'
)

# Get the current date and time
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS

# Train the model using your existing data loaders
# results = train_model(
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     device=device,
#     epochs=args.epochs,
#     lr=args.lr,
#     weight_decay=args.weight_decay
# )

# Update the training call to include the mixup_alpha parameter
results = train_model_mixup(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    epochs=args.epochs,
    lr=args.lr,
    weight_decay=args.weight_decay,
    mixup_alpha=0.4  # Add this parameter - set to 0 to disable MixUp
)

# Save model with date and time in the filename
model_path = os.path.join("models", f"cnnbilstm-window-{args.window_length}-{args.note}-{current_time}.pth")
torch.save(model.state_dict(), model_path)

print("Training completed!")
print(f"Final validation CER: {results['final_val_cer']:.4f}")

# After training, you can finish the WandB run
wandb.finish()
# %%
