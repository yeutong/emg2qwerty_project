# %%
import torch
import matplotlib.pyplot as plt
from emg2qwerty.transforms import LogSpectrogram
import numpy as np
from pathlib import Path

# %%
import h5py

emg2qwerty_path = Path(__file__).parent.parent / "emg2qwerty"
data_path = emg2qwerty_path / "data/2021-06-02-1622679967-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5"

with h5py.File(data_path, 'r') as f:
    # for k, v in f['emg2qwerty'].attrs.items():
    #     print(k)
    #     print(v)
    #     print('======')

    data = f['emg2qwerty']['timeseries'][:]

# %%

# Assuming 'data' is your NumPy structured array
emg_right = data['emg_right']  # Shape: (num_samples, 16)
time = data['time']            # Shape: (num_samples,)
emg_left = data['emg_left']    # Shape: (num_samples, 16)

# %%

# # Create a simple synthetic EMG signal for visualization
# def create_sample_emg(time_steps=1000):
#     t = np.linspace(0, 1, time_steps)
#     # Simulate an EMG burst
#     emg = np.sin(2 * np.pi * 10 * t) * np.exp(-(t - 0.5)**2 / 0.1**2)
#     emg += np.random.normal(0, 0.1, time_steps)  # Add some noise
#     return emg

# # Create sample data
# time_steps = 1000
# emg_left = create_sample_emg(time_steps)
# emg_right = create_sample_emg(time_steps)

# # Convert to torch tensor with shape (T, 2, 16)
# # Simulate 16 channels for each of left and right EMG
# emg_data = torch.stack([
#     torch.from_numpy(emg_left).float().unsqueeze(1).repeat(1, 16),
#     torch.from_numpy(emg_right).float().unsqueeze(1).repeat(1, 16)
# ], dim=1)

emg_data = torch.from_numpy(emg_right[2000:3000, :]).float()

# Create and apply LogSpectrogram
transform = LogSpectrogram(n_fft=64, hop_length=16)
spectrogram = transform(emg_data)

# Plotting
plt.figure(figsize=(15, 10))

# Plot original EMG signal
plt.subplot(2, 1, 1)
plt.plot(emg_data[:, 0].numpy())  # Plot first channel of left EMG
plt.title('Original EMG Signal (Left, Channel 0)')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Plot spectrograms
plt.subplot(2, 1, 2)
plt.imshow(spectrogram[:, 0].numpy().T, 
           aspect='auto', 
           origin='lower',
           cmap='viridis')
plt.title('Log Spectrogram (Left, Channel 0)')
plt.xlabel('Time')
plt.ylabel('Frequency Bin')
plt.colorbar(label='Log Power')


plt.tight_layout()
plt.show()

# Print shapes
print("Original EMG data shape:", emg_data.shape)
print("Spectrogram shape:", spectrogram.shape)

# %%
from data import create_transforms
from emg2qwerty.data import WindowedEMGDataset
from config import load_config
train_transform, eval_transform = create_transforms()

# Create datasets
train_datasets = []
val_datasets = []
test_datasets = []

use_keystroke_augmentation=False

batch_size = 128
num_workers = 2

emg2qwerty_path = Path(__file__).parent.parent / "emg2qwerty"

# Load config
config_path = emg2qwerty_path / "config/user/single_user.yaml"
cfg = load_config(config_path)

# Load data
data_path = emg2qwerty_path / "data"

# Create training datasets with augmentation
for session_info in cfg['dataset']['train']:
    base_dataset = WindowedEMGDataset(
        hdf5_path=data_path / f"{session_info['session']}.hdf5",
        window_length=2000,
        stride=1000,
        padding=(200, 200),
        jitter=True,
        transform=train_transform
    )

    train_datasets.append(base_dataset)
# %%
train_dataset_one_ori = train_datasets[0]
# %%
emg, labels = train_dataset_one_ori.__getitem__(0)
# %%
