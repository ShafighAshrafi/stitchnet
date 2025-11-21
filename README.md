# Reproducibility Guide

This document provides step-by-step instructions to reproduce the results obtained in this StitchNet project.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Workflow A: Implementation Without Fine-tuning](#workflow-a-implementation-without-fine-tuning)
5. [Workflow B: Implementation With Fine-tuning](#workflow-b-implementation-with-fine-tuning)
6. [Key Parameters](#key-parameters)
7. [Expected Results](#expected-results)

---

## Prerequisites

### Hardware Requirements

- GPU with CUDA support (recommended for training and inference)
- Sufficient disk space for models and results (several GB)
- RAM: 16GB+ recommended

### Software Requirements

- Python 3.10 or 3.11
- CUDA 11.8 (for GPU support)
- Jupyter Notebook or JupyterLab

---

## Environment Setup

### 1. Install Dependencies

Install all required packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**Key dependencies:**

- PyTorch 2.2.2+cu118 (with CUDA 11.8 support)
- ONNX 1.15.0
- ONNX Runtime GPU 1.17.1
- Torchvision 0.17.2+cu118
- Other dependencies as listed in `requirements.txt`

### 2. Verify Installation

Ensure CUDA is available:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## Dataset Preparation

### 1. Prepare Dataset Structure

The dataset should be organized in the following structure:

```
src/dataset/main_dataset/
├── train/
│   ├── class0/
│   ├── class1/
│   └── class2/
└── test/
    ├── class0/
    ├── class1/
    └── class2/
```

### 2. Run Data Preparation

Execute the data preparation notebook:

```bash
jupyter notebook base_computations/1_prepare_data.ipynb
```

Or run the Python script:

```bash
python generate_data.py
```

This will:

- Read worst-case images
- Create train/test splits

---

## Workflow A: Implementation Without Fine-tuning

This workflow generates stitched networks from pre-trained models without initial fine-tuning.

### Step 1: Download Pre-trained Models

**Option A: Using Jupyter Notebook**

```bash
jupyter notebook base_computations/2_download_models-Copy1.ipynb
```

**Option B: Using Python Script**

```bash
python download_models.py
```

This downloads and converts the following models to ONNX format:

- ResNet50
- AlexNet
- DenseNet121
- MobileNet V3 Small
- VGG16

Models will be saved in `_models/` directory.

### Step 2: Generate Fragments

Execute the fragment generation notebook:

```bash
jupyter notebook implementation_without_fine_tuning/1_generate_fragments.ipynb
```

**What this does:**

- Loads ONNX models from `_models/`
- Splits each model into fragments at convolutional layer boundaries
- Saves fragments to `_results_without_finetune/fragments/`

**Expected output:**

- Multiple fragments per model (e.g., 8 for AlexNet, 5 for MobileNet, 13 for DenseNet, 6 for VGG16, 16 for ResNet50)

### Step 3: Generate Stitch Networks

Execute the stitching notebook:

```bash
jupyter notebook implementation_without_fine_tuning/2_generate_stitch_networks.ipynb
```

**Key Parameters (set in the notebook):**

```python
random.seed(51)
np.random.seed(24)
torch.manual_seed(77)

K = 5
STITCH_BATCH_SIZE = 32
MAX_DEPTH = 16
THRESOULD = 0
TOTAL_THRESOULD = 0.5
EVAL_BATCH_SIZE = 64
```

**What this does:**

- Creates score mapper for fragments
- Generates stitched networks by combining fragments from different models
- Evaluates each generated network
- Saves results to `_results_without_finetune/{RESULT_NAME}/`

**Result naming format:**
`{timestamp}_result_BS_{STITCH_BATCH_SIZE}_MD_{MAX_DEPTH}_T_{THRESOULD}_TT_{TOTAL_THRESOULD}_K_{K}`

### Step 4: Evaluate Stitch Networks

Execute the evaluation notebook:

```bash
jupyter notebook implementation_without_fine_tuning/3_evaluate_stitch_networks.ipynb
```

**What this does:**

- Evaluates accuracy of all generated stitch networks
- Calculates MACs (Multiply-Accumulate operations) and parameters
- Generates visualizations of network structures
- Saves evaluation results

## Workflow B: Implementation With Fine-tuning

This workflow fine-tunes original models first, then generates stitched networks.

### Step 1: Fine-tune Original Models

Execute the fine-tuning notebook:

```bash
jupyter notebook implementation_with_fine_tuning/1_finetune_models.ipynb
```

**Parameters:**

```python
batch_size = 64  # 32 for DenseNet121
num_epochs = 3
feature_extract = False  # Full fine-tuning
```

**What this does:**

- Fine-tunes each pre-trained model on the target dataset
- Saves training history and plots
- Exports fine-tuned models to ONNX format
- Models saved to `_results_with_finetune/finetune/{model_name}/`

### Step 2: Generate Fragments from Fine-tuned Models

Execute the fragment generation notebook:

```bash
jupyter notebook implementation_with_fine_tuning/2_generate_fragments.ipynb
```

**What this does:**

- Loads fine-tuned ONNX models
- Splits models into fragments
- Saves fragments to `_results_with_finetune/fragments/`

### Step 3: Generate Stitch Networks

Execute the stitching notebook:

```bash
jupyter notebook implementation_with_fine_tuning/3_generate_stitch_networks.ipynb
```

**Parameters (same as Workflow A):**

```python
random.seed(51)
np.random.seed(24)
torch.manual_seed(77)

K = 5
STITCH_BATCH_SIZE = 32
MAX_DEPTH = 16
THRESOULD = 0
TOTAL_THRESOULD = 0.5
EVAL_BATCH_SIZE = 64
```

### Step 4: Evaluate Stitch Networks

Execute the evaluation notebook:

```bash
jupyter notebook implementation_with_fine_tuning/4_evaluate_stitch_networks.ipynb
```

### Step 5: Fine-tune Stitch Networks

Execute the fine-tuning notebook:

```bash
jupyter notebook implementation_with_fine_tuning/6_finetune_stitch_models.ipynb
```

**Parameters:**

```python
batch_size = 32
val_batch_size = 64
num_epochs = 10  # or adjust as needed
```

---

## Additional Components

### Autoencoder Training (Optional)

For representation learning, train an autoencoder:

```bash
jupyter notebook base_computations/4_train_autoencoder.ipynb
```

**Parameters:**

```python
max_dimension = 4096
encoding_dim = 50
learning_rate = 0.01
num_epochs = 1
```

**Output:** `_models/autoencoder.pt`

### RNN Training (Optional)

Train an RNN for accuracy prediction:

```bash
jupyter notebook base_computations/5_train_rnn_with_autoencoder.ipynb
```

**Parameters:**

```python
input_size = 4096 * 256
hidden_size = 64
output_size = 1
num_layers = 1
learning_rate = 0.01
num_epochs = 1
```

**Output:** `_models/rnn_autoencoder.pt`

---

## Key Parameters

### Reproducibility Seeds

```python
random.seed(51)      # Python random
np.random.seed(24)   # NumPy random
torch.manual_seed(77)  # PyTorch random
```

### Stitching Parameters

- **K**: Number of top fragments to consider (default: 5)
- **MAX_DEPTH**: Maximum depth of stitched network (default: 16)
- **THRESOULD**: Fragment selection threshold (default: 0)
- **TOTAL_THRESOULD**: Total score threshold (default: 0.5)
- **STITCH_BATCH_SIZE**: Batch size for stitching (default: 32)
- **EVAL_BATCH_SIZE**: Batch size for evaluation (default: 64)

### Training Parameters

- **Learning Rate**: 0.001 (for fine-tuning), 0.01 (for autoencoder/RNN)
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss (with class weights for imbalanced data)
- **Epochs**: 3 (initial fine-tuning), 3-10 (stitch network fine-tuning)

### Model-Specific Settings

- **DenseNet121**: Uses batch_size=32 and num_epochs=3 due to memory constraints
- **Other models**: Use batch_size=64 and num_epochs=3

---

## Expected Results

### Directory Structure

After running the workflows, you should see:

```
_results_without_finetune/
├── fragments/              # Model fragments
├── {timestamp}_result_*/   # Generated stitch networks
│   ├── net000/
│   ├── net001/
│   └── ...
├── finetune_3/            # Fine-tuned stitch networks (3 epochs)
├── finetune_5/            # Fine-tuned stitch networks (5 epochs)
└── finetune_10/           # Fine-tuned stitch networks (10 epochs)

_results_with_finetune/
├── finetune/              # Fine-tuned original models
│   ├── resnet50/
│   ├── alexnet/
│   └── ...
├── fragments/             # Fragments from fine-tuned models
├── {timestamp}_result_*/  # Generated stitch networks
└── evaluation_after_finetuning/
    └── finetune_{epochs}/ # Fine-tuned stitch networks
```

### Result Files

Each network directory contains:

- `{accuracy}.txt`: Final accuracy value
- `model_ft.onnx`: Fine-tuned ONNX model (if fine-tuned)
- `accuracy.png`: Training accuracy curve
- `loss.png`: Training loss curve

### Evaluation Metrics

Results include:

- **Accuracy**: Validation and training accuracy
- **MACs**: Multiply-Accumulate operations
- **Parameters**: Number of model parameters
- **Score**: Network score from scoring function

## Notes

- Results may vary slightly due to GPU non-determinism, even with seeds set
- Processing time depends on GPU capability and dataset size
- Some models (especially DenseNet121) require more memory
- The project uses class-weighted loss to handle imbalanced datasets
- All models are evaluated on ImageNet pre-trained weights before fine-tuning

---

## Citation

If you use this code, please cite the original StitchNet paper and this implementation.

---

## Contact

For questions or issues regarding reproducibility, please create an issue in the repository.
