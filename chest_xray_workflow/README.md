# Chest X-Ray Pneumonia Dataset - StitchNet Workflow

This directory contains the complete workflow for applying StitchNet to the Chest X-Ray Pneumonia dataset from Kaggle. This addresses the cross-domain transfer limitation mentioned in the original evaluation.

## Dataset

**Chest X-Ray Pneumonia Dataset**
- Source: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Classes: 2 (NORMAL, PNEUMONIA)
- Image size: 224x224
- Format: RGB images

## Workflow Overview

The workflow consists of 7 main steps:

1. **Prepare Data** (`1_prepare_data.ipynb`) - Download and prepare the chest X-ray dataset
2. **Download Models** (`2_download_models.ipynb`) - Download pre-trained ImageNet models
3. **Fine-tune Models** (`3_finetune_models.ipynb`) - Fine-tune models on chest X-ray dataset (2 classes)
4. **Generate Fragments** (`4_generate_fragments.ipynb`) - Split models into fragments
5. **Generate Stitch Networks** (`5_generate_stitch_networks.ipynb`) - Create stitched networks
6. **Evaluate Stitch Networks** (`6_evaluate_stitch_networks.ipynb`) - Evaluate generated networks
7. **Fine-tune Stitch Models** (`7_finetune_stitch_models.ipynb`) - Fine-tune stitched networks

## Quick Start

### Step 1: Download the Dataset

**Option A: Using Kaggle API (Recommended)**
```bash
# Install kaggle if not already installed
pip install kaggle

# Set up Kaggle credentials (see download_chest_xray_dataset.py for instructions)
python download_chest_xray_dataset.py
```

**Option B: Manual Download**
1. Go to https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Download the dataset
3. Extract to `chest_xray_dataset/` folder in project root

### Step 2: Prepare the Dataset

```bash
python prepare_chest_xray_data.py
```

Or use the notebook:
```bash
jupyter notebook chest_xray_workflow/1_prepare_data.ipynb
```

### Step 3: Run the Workflow

Execute notebooks in order:

1. `1_prepare_data.ipynb` - Prepare dataset
2. `2_download_models.ipynb` - Download pre-trained models
3. `3_finetune_models.ipynb` - Fine-tune models (2 classes)
4. `4_generate_fragments.ipynb` - Generate fragments
5. `5_generate_stitch_networks.ipynb` - Generate stitched networks
6. `6_evaluate_stitch_networks.ipynb` - Evaluate networks
7. `7_finetune_stitch_models.ipynb` - Fine-tune stitched networks

## Key Differences from Original Workflow

### Dataset
- **Original**: 3 classes (normal, drusen, cnv) - Retinal OCT
- **Chest X-Ray**: 2 classes (NORMAL, PNEUMONIA) - Chest X-ray images

### Model Configuration
- All models are configured for **2 classes** instead of 3
- Fine-tuning functions use `num_classes=2`
- Dataset loader: `load_dataset_chest_xray` instead of `load_dataset`

### File Paths
- Models: `_models_chest_xray/` instead of `_models/`
- Results: `_results_chest_xray/` instead of `_results_without_finetune/` or `_results_with_finetune/`

## Directory Structure

After running the workflow:

```
_results_chest_xray/
├── finetune/              # Fine-tuned original models
│   ├── resnet50/
│   ├── alexnet/
│   └── ...
├── fragments/             # Model fragments
│   ├── net000/
│   └── ...
├── {timestamp}_result_*/  # Generated stitch networks
│   ├── net000/
│   ├── net001/
│   └── ...
├── original/              # Evaluation of original models
└── evaluation_after_finetuning/
    └── finetune_{epochs}/  # Fine-tuned stitch networks
```

## Parameters

### Stitching Parameters
- **K**: 5 (number of top fragments to consider)
- **MAX_DEPTH**: 16 (maximum depth of stitched network)
- **THRESOULD**: 0 (fragment selection threshold)
- **TOTAL_THRESOULD**: 0.5 (total score threshold)
- **STITCH_BATCH_SIZE**: 32
- **EVAL_BATCH_SIZE**: 64

### Training Parameters
- **Learning Rate**: 0.001 (for fine-tuning)
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss (with class weights for imbalanced data)
- **Epochs**: 3 (initial fine-tuning), 10 (stitch network fine-tuning)

### Reproducibility Seeds
```python
random.seed(51)
np.random.seed(24)
torch.manual_seed(77)
```

## Expected Results

### Dataset Statistics
- Training samples: ~5,000+ (varies by split)
- Test samples: ~600+ (varies by split)
- Classes: NORMAL, PNEUMONIA

### Model Performance
- Fine-tuned models should achieve >90% accuracy on chest X-ray dataset
- Stitched networks will have varying performance based on fragment combinations
- Best models will be saved with their accuracy scores

## Troubleshooting

### Common Issues

1. **Dataset Not Found**
   - Ensure dataset is downloaded and extracted to `chest_xray_dataset/`
   - Update path in `1_prepare_data.ipynb` if needed

2. **CUDA Out of Memory**
   - Reduce batch sizes (especially for DenseNet121)
   - Process models one at a time
   - Use CPU for smaller models

3. **Class Mismatch Errors**
   - Ensure all code uses `num_classes=2`
   - Check that `load_dataset_chest_xray` is used instead of `load_dataset`

4. **Import Errors**
   - Ensure you're running notebooks from the `chest_xray_workflow/` directory
   - Check that `sys.path.insert(0, os.path.abspath('..'))` is in each notebook

## Notes

- This workflow demonstrates cross-domain transfer from ImageNet (natural images) to medical imaging (chest X-rays)
- The 2-class classification (NORMAL vs PNEUMONIA) is a binary classification task
- Results may vary due to GPU non-determinism, even with seeds set
- The chest X-ray dataset is typically imbalanced (more PNEUMONIA cases), so class weights are used in loss function

## Citation

If you use this code, please cite:
- The original StitchNet paper
- The Chest X-Ray Pneumonia dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

