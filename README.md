# ML Trainer Framework

A modular PyTorch-based machine learning trainer designed for both image and tabular data. This framework supports classification and regression tasks, with built-in support for logging, early stopping, checkpointing, and multi-model/multi-dataset experimentation.

---

## 📦 Project Structure

```
project/
├── checkpoints/                # Saved model checkpoints
├── configs/                    # YAML config files for models, training, data
│   └── default.yaml
├── data/                       # Dataset loaders and preprocessing
│   ├── dataloader.py
│   └── preprocess.py
├── engine/                     # Core training/validation/testing logic
│   ├── trainer.py
│   ├── validator.py
│   └── tester.py
├── logs/                       # TensorBoard and CSV log outputs
├── models/                     # Model definitions
│   ├── mlp.py                  # For tabular data
│   └── cnn.py                  # For image data
├── utils/                      # Helper utilities
│   ├── logger.py
│   ├── metrics.py
│   ├── early_stopping.py
│   └── checkpoint.py
├── main.py                     # Entry point for training
└── firebase_utils.py           # (Unused, placeholder for optional Firebase integration)
```

---

## 🚀 Features

- ✅ PyTorch-based training loop
- ✅ Supports both image and tabular datasets
- ✅ Classification and regression support
- ✅ Configurable via YAML config files
- ✅ Checkpoint saving & loading
- ✅ Early stopping
- ✅ Training/validation/testing separation
- ✅ CSV and TensorBoard logging
- ✅ CUDA and MPS support (GPU acceleration)
- ❌ No Firebase integration (yet)
- ❌ No multi-GPU or mixed precision (by design)

---

## 🧩 Usage

### 1. Install Dependencies
```bash
pip install torch torchvision pandas scikit-learn tensorboard pyyaml
```

### 2. Prepare Your Config
Edit or create a config file under `configs/`, e.g.:
```yaml
model: "mlp"
task: "classification"
dataset: "your_dataset"
epochs: 50
batch_size: 64
learning_rate: 0.001
device: "cuda"  # or "mps" or "cpu"
```

### 3. Run Training
```bash
python main.py --config configs/default.yaml
```

---

## 🔧 Customization

- **New models**: Add a new file under `models/` and register it in your config.
- **New datasets**: Implement a DataLoader in `data/dataloader.py`.
- **Add metrics**: Extend `utils/metrics.py`.
- **Logger format**: Customize `utils/logger.py`.

---

## 📊 Logging

- **TensorBoard**: logs saved in `logs/tensorboard/`
- **CSV**: logs saved in `logs/csv/`

Start TensorBoard with:
```bash
tensorboard --logdir logs/tensorboard/
```

---

## 🧠 Notes

- Designed for solo or small team ML competitions and practical projects.
- No dependency on third-party experiment platforms.
- Firebase placeholder exists but not used.

---

## 📄 License

MIT License
