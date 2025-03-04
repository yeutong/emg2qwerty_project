# %%
import h5py
# %%
data_path = '../data/89335547/2021-06-02-1622679967-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5'
f = h5py.File(data_path, 'r')
# %%
for k, v in f['emg2qwerty'].attrs.items():
    print(k)
    print(v)

    print('======')

# %%
data = f['emg2qwerty']['timeseries'][:]

# %%

# Assuming 'data' is your NumPy structured array
emg_right = data['emg_right']  # Shape: (num_samples, 16)
time = data['time']            # Shape: (num_samples,)
emg_left = data['emg_left']    # Shape: (num_samples, 16)

# %%
import matplotlib.pyplot as plt

# Plot the first channel of emg_right and emg_left for the first 100 samples
plt.figure(figsize=(12, 6))

# Plot emg_right
plt.subplot(2, 1, 1)
plt.plot(time[:100], emg_right[:100, 0], label='EMG Right - Channel 1')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('EMG Right - Channel 1')
plt.legend()

# Plot emg_left
plt.subplot(2, 1, 2)
plt.plot(time[:100], emg_left[:100, 0], label='EMG Left - Channel 1', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('EMG Left - Channel 1')
plt.legend()

plt.tight_layout()
plt.show()


# %%
f.close()
# %%
