# Conformer-based EMG-to-Text Decoding

This repository extends the original EMG-to-text decoding project from [joe-lin-tech/emg2qwerty](https://github.com/joe-lin-tech/emg2qwerty). The project explores the use of a **Conformer-based model** for **surface electromyography (sEMG) signal-based text decoding**, analyzing various factors that impact **Character Error Rate (CER)**.

## 📌 Setup Instructions

### 1️⃣ **Clone the Original Repository**
The professor provided the following GitHub repository as the base project:

```bash
git clone https://github.com/joe-lin-tech/emg2qwerty.git
cd emg2qwerty
```

### 2️⃣ **Follow Colab Setup**
The original repository contains a **Colab setup notebook** that configures the environment. Follow the instructions inside:

- **From the original repo:** `emg2qwerty/Colab_setup.ipynb`
- **Or from this project:** `emg2qwerty_project/emg2qwerty/Colab_setup.ipynb`

### 3️⃣ **Add the Conformer Notebooks**
After cloning the repository, place the following **five Jupyter notebooks** inside the `emg2qwerty/` folder. This ensures they can access models, scripts, and dataset configurations from:
- `emg2qwerty/models/`
- `emg2qwerty/scripts/`
- `emg2qwerty/emg2qwerty/`
- `emg2qwerty/config/`
- `emg2qwerty/data/` (Follow Colab_setup.ipynb for data placement instructions.)

📂 **Directory Structure:**
```
emg2qwerty/
│── models/           # Model implementations
│── scripts/          # Helper scripts and data loading
│── emg2qwerty/       # Core functions
│── config/           # Configuration files
│── data/             # Dataset storage (as per Colab setup instructions)
│── ConformerCode.ipynb          # Main Conformer training and evaluation
│── ConformerCodeChan.ipynb      # Electrode channels vs. CER
│── ConformerCodeData.ipynb      # Training data size vs. CER
│── ConformerCodeGuassian.ipynb  # Gaussian noise augmentation
│── ConformerCodeSamp.ipynb      # Sampling rate vs. CER
```

## 🚀 Running the Conformer Model

### **Run the Main Model**
```bash
jupyter notebook ConformerCode.ipynb
```

### **Experiment Variations:**
Each notebook explores a different factor affecting CER:
1. **Electrode Channels (`ConformerCodeChan.ipynb`)** - Varying number of channels
2. **Training Data Size (`ConformerCodeData.ipynb`)** - Effect of dataset size
3. **Gaussian Noise (`ConformerCodeGuassian.ipynb`)** - Robustness to noise
4. **Sampling Rate (`ConformerCodeSamp.ipynb`)** - Impact of different frequencies

## 🔬 Purpose of the Project
- **Perform a comparative analysis of the Conformer model** for sEMG-based text decoding.
- **Evaluate Conformer's impact on CER** under different experimental conditions.
- **Compare Conformer’s performance** against the existing TDS baseline model to assess its effectiveness in sEMG-based decoding.
- **Analyze key factors** (electrodes, noise, sampling) influencing CER to optimize model performance.

## 📌 Additional Notes
- The models and scripts remain the same as the base project.
- Only Jupyter notebooks are added for analysis.
- Follow `Colab_setup.ipynb` from the original repo for dependencies.

This ensures **full compatibility** with the existing **emg2qwerty** project while enabling further experiments with Conformer-based models.

