
---

## 🧩 Files Explained

### 1. `Date.py`
- Adds a **date column** to the dataset (`dataset_machine.csv`).
- Date range: `1998-01-10` to `2025-05-27` (daily records).
- Appends this column to the original dataset.

---

### 2. `scale.py`
- **Normalizes sensor columns** (like `air_temp`, `process_temp`, `torque`, etc.) to the 0–1 range using **MinMaxScaler**.
- Creates **sequential time-series data** for training using numerical features only.
- Converts the data to **PyTorch tensors**.
- Defines an **LSTM Autoencoder**:
  - **Encoder**: Compresses the input sequence.
  - **Decoder**: Reconstructs the original sequence from encoded features.
- Initializes the model and trains it for **20 epochs**.
- Saves the trained model weights to `model_weights.pth`.

---

### 3. `train.py`
- Imports the `LSTMAutoencoder` from `scale.py`.
- Defines the same **input size** and **hidden size** used during training.
- Loads the model weights from `model_weights.pth`.
- Computes the **reconstruction error** on the input sequence.
- Converts reconstruction error to a NumPy array.
- Computes **threshold** using:
  - `threshold = mean + k * std`
  - `k = 3` for strict detection (fewer false positives), or `k = 1 or 2` for lenient detection.
- Prints the **detected anomalies**.
- **Plots** the reconstruction error vs threshold.

---

## 📦 Dataset

- **File:** `dataset_machine.csv`
- Contains time-series sensor data from machines.
- Sensor columns may include: `air_temperature`, `process_temperature`, `torque`, `tool_wear`, etc.
- One record per day.

---

## 🚀 How to Run

### Step 1: Add Dates to Dataset
```bash
python Date.py
