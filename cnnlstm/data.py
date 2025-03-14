from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision.transforms import Compose
from pathlib import Path
from torch.utils.data import Dataset

from emg2qwerty.data import WindowedEMGDataset
from emg2qwerty.transforms import ToTensor, LogSpectrogram, SpecAugment, RandomBandRotation
from augment import KeystrokeTransitionAugmentation

def create_transforms():
    """Create training and evaluation transforms"""
    train_transform = Compose([
        ToTensor(),
        LogSpectrogram(),
        SpecAugment(
            n_time_masks=2,
            time_mask_param=20,
            n_freq_masks=2,
            freq_mask_param=2,

        ),
        RandomBandRotation()
    ])
    
    eval_transform = Compose([
        ToTensor(),
        LogSpectrogram()
    ])
    
    return train_transform, eval_transform

def create_dataloaders(cfg, data_dir, batch_size=128, num_workers=2, use_keystroke_augmentation=False, downsample_rate=1, num_sessions=16, channel_drop_rate=0):
    """Create train, validation, and test dataloaders"""
    train_transform, eval_transform = create_transforms()
    
    # Create datasets
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    # Create training datasets with augmentation
    for session_idx, session_info in enumerate(cfg['dataset']['train']):
        # Exp 2: how much data to use
        if session_idx >= num_sessions:
            break

        base_dataset = WindowedEMGDataset(
            hdf5_path=data_dir / f"{session_info['session']}.hdf5",
            window_length=2000,
            stride=1000,
            padding=(200, 200),
            jitter=True,
            transform=train_transform,
            downsample_rate=downsample_rate,
            channel_drop_rate=channel_drop_rate
        )
        # Wrap with keystroke augmentation
        if use_keystroke_augmentation:
            augmented_dataset = KeystrokeAugmentedDataset(base_dataset, apply_augmentation=True)
            train_datasets.append(augmented_dataset)
        else:
            train_datasets.append(base_dataset)
    
    # Create validation and test datasets (no keystroke augmentation)
    val_datasets = []
    for session_info in cfg['dataset']['val']:
        session_id = session_info['session']
        file_path = data_dir / f"{session_id}.hdf5"

        dataset = WindowedEMGDataset(
            hdf5_path=file_path,
            window_length=2000,
            stride=1000,
            padding=(200, 200),
            jitter=False,  # No jitter for validation
            transform=eval_transform,
            downsample_rate=downsample_rate,
            channel_drop_rate=channel_drop_rate
        )
        val_datasets.append(dataset)

    test_datasets = []
    for session_info in cfg['dataset']['test']:
        session_id = session_info['session']
        file_path = data_dir / f"{session_id}.hdf5"

        dataset = WindowedEMGDataset(
            hdf5_path=file_path,
            window_length=2000,
            stride=1000,
            padding=(200, 200),
            jitter=False,  # No jitter for testing
            transform=eval_transform,
            downsample_rate=downsample_rate,
            channel_drop_rate=channel_drop_rate
        )
        test_datasets.append(dataset)
    
    # Combine datasets
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    test_dataset = ConcatDataset(test_datasets)
    
    collate_fn = KeystrokeAugmentedDataset.collate if use_keystroke_augmentation else WindowedEMGDataset.collate
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

class KeystrokeAugmentedDataset(Dataset):
    """Wrapper dataset that applies keystroke augmentation to WindowedEMGDataset."""
    
    def __init__(self, base_dataset, apply_augmentation=True):
        """
        Args:
            base_dataset: The base WindowedEMGDataset
            apply_augmentation: Whether to apply the keystroke augmentation
        """
        self.base_dataset = base_dataset
        self.augmentation = KeystrokeTransitionAugmentation(
            substitution_prob=0.15,
            max_substitutions=3,
            only_adjacent=True
        ) if apply_augmentation else None
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get sample from base dataset
        inputs, targets = self.base_dataset[idx]
        
        # Apply keystroke augmentation if enabled
        if self.augmentation is not None:
            inputs, targets = self.augmentation((inputs, targets))
            
        return inputs, targets
    
    @staticmethod
    def collate(batch):
        """Use the same collate function as WindowedEMGDataset"""
        from emg2qwerty.data import WindowedEMGDataset
        return WindowedEMGDataset.collate(batch)
