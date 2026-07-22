# NIH Chest X-ray Pipeline: User Commands Cheat-Sheet

Use this quick-reference file to easily copy and paste the commands needed to train, fine-tune, test, and run predictions for any of the supported architectures.

---

## 📌 Section 1: DenseNet-121 (Your Current Gold-Standard Model)

These commands are configured specifically for the `densenet121` model, which uses the pre-trained weights from the `pre-trained DenseNet121 small` directory.

### 1. Fine-Tune/Train the Model
Runs fine-tuning for 15 epochs on your GPU with mixed precision (`--use_amp`), balanced weighting (`--damp_weights`), lighting augmentations (`--augment_brightness_contrast`), and backbone freezing on Epoch 1:
```bash
python src/train.py --model_name densenet121 --use_amp --damp_weights --augment_brightness_contrast --freeze_epochs 1 --run_name densenet121_best_accuracy_run
```

*To resume training if it gets interrupted, simply add `--resume` to the end:*
```bash
python src/train.py --model_name densenet121 --use_amp --damp_weights --augment_brightness_contrast --freeze_epochs 1 --run_name densenet121_best_accuracy_run --resume
```

### 2. Test / Evaluate the Model
Evaluates your best checkpoint on the test set of 25,596 images and prints metrics:
```bash
python src/test.py --model_name densenet121 --checkpoint_path checkpoints/densenet121_best_accuracy_run/best_model_auc.pth --output_report evaluation_report_optimized.txt
```
*(Simplest shorthand version: `python src/test.py --checkpoint_path checkpoints/densenet121_best_accuracy_run/best_model_auc.pth`)*

### 3. Run Predictions on a Single X-ray Image
Pass any X-ray image path to see a sorted list of the 14 disease probabilities:
```bash
python src/predict.py --model_name densenet121 --checkpoint_path checkpoints/densenet121_best_accuracy_run/best_model_auc.pth --image_path images/00000001_000.png --threshold 0.20
```

---

## 📌 Section 2: Other Supported Architectures

These sections contain the copy-paste commands if you decide to train or test other models.

---

### A. CheXNet (DenseNet-121 Pre-trained on X-rays)
*Highly recommended starting point as the base weights are already specialized in radiological scans.*

#### 1. Fine-Tune/Train:
```bash
python src/train.py --model_name chexnet --use_amp --damp_weights --augment_brightness_contrast --freeze_epochs 1 --run_name chexnet_run
```
#### 2. Resume Interrupted Training:
```bash
python src/train.py --model_name chexnet --use_amp --damp_weights --augment_brightness_contrast --freeze_epochs 1 --run_name chexnet_run --resume
```
#### 3. Test/Evaluate:
```bash
python src/test.py --model_name chexnet --checkpoint_path checkpoints/chexnet_run/best_model_auc.pth --output_report evaluation_report_chexnet.txt
```
#### 4. Run Predictions:
```bash
python src/predict.py --model_name chexnet --checkpoint_path checkpoints/chexnet_run/best_model_auc.pth --image_path images/00000001_000.png --threshold 0.20
```

---

### B. EfficientNet-B4 (Medium Convolutional Model)
*Lighter, newer, and highly resource-efficient architecture.*

#### 1. Fine-Tune/Train:
```bash
python src/train.py --model_name efficientnet_b4 --use_amp --damp_weights --augment_brightness_contrast --freeze_epochs 1 --run_name effnet_run
```
#### 2. Resume Interrupted Training:
```bash
python src/train.py --model_name efficientnet_b4 --use_amp --damp_weights --augment_brightness_contrast --freeze_epochs 1 --run_name effnet_run --resume
```
#### 3. Test/Evaluate:
```bash
python src/test.py --model_name efficientnet_b4 --checkpoint_path checkpoints/effnet_run/best_model_auc.pth --output_report evaluation_report_effnet.txt
```
#### 4. Run Predictions:
```bash
python src/predict.py --model_name efficientnet_b4 --checkpoint_path checkpoints/effnet_run/best_model_auc.pth --image_path images/00000001_000.png --threshold 0.20
```

---

### C. Swin-T (Medium Vision Transformer Model)
*Uses local self-attention windows to compare different regions of the lungs for diagnostic context.*

#### 1. Fine-Tune/Train:
```bash
python src/train.py --model_name swin_t --use_amp --damp_weights --augment_brightness_contrast --freeze_epochs 1 --run_name swin_run
```
#### 2. Resume Interrupted Training:
```bash
python src/train.py --model_name swin_t --use_amp --damp_weights --augment_brightness_contrast --freeze_epochs 1 --run_name swin_run --resume
```
#### 3. Test/Evaluate:
```bash
python src/test.py --model_name swin_t --checkpoint_path checkpoints/swin_run/best_model_auc.pth --output_report evaluation_report_swin.txt
```
#### 4. Run Predictions:
```bash
python src/predict.py --model_name swin_t --checkpoint_path checkpoints/swin_run/best_model_auc.pth --image_path images/00000001_000.png --threshold 0.20
```

---

### D. ResNet-50 (Medium Standard Benchmark Model)
*Standard baseline benchmark model.*

#### 1. Fine-Tune/Train:
```bash
python src/train.py --model_name resnet50 --use_amp --damp_weights --augment_brightness_contrast --freeze_epochs 1 --run_name resnet50_run
```
#### 2. Resume Interrupted Training:
```bash
python src/train.py --model_name resnet50 --use_amp --damp_weights --augment_brightness_contrast --freeze_epochs 1 --run_name resnet50_run --resume
```
#### 3. Test/Evaluate:
```bash
python src/test.py --model_name resnet50 --checkpoint_path checkpoints/resnet50_run/best_model_auc.pth --output_report evaluation_report_resnet50.txt
```
#### 4. Run Predictions:
```bash
python src/predict.py --model_name resnet50 --checkpoint_path checkpoints/resnet50_run/best_model_auc.pth --image_path images/00000001_000.png --threshold 0.20
```

---

## 📌 Section 3: Model Visualizations — `visualize_model.py`

This script generates **10 rich visualizations** for any trained model and saves them automatically to the correct `info/<model>-test-output/` folder.

**Charts generated:**
| # | File Name | Description |
| :---: | :--- | :--- |
| 01 | `01_roc_curves.png` | ROC curve for all 14 diseases on one figure |
| 02 | `02_auc_per_disease.png` | Horizontal AUC bar chart, sorted best→worst |
| 03 | `03_precision_recall_f1.png` | Grouped bar: Precision / Recall / F1 per disease |
| 04 | `04_f1_heatmap.png` | Color heatmap of P / R / F1 scores |
| 05 | `05_confidence_distributions.png` | Violin plots: model confidence for positive vs negative cases |
| 06 | `06_top_bottom_auc.png` | Top-5 and Bottom-5 diseases by AUC with gap vs mean |
| 07 | `07_training_curves.png` | Loss & AUC curves per epoch (if history embedded in checkpoint) |
| 08 | `08_confusion_matrix_grid.png` | Confusion matrices for the 4 most common diseases |
| 09 | `09_threshold_sensitivity.png` | F1 vs decision threshold for best/worst diseases |
| 10 | `10_model_summary_card.png` | Full stats card: mean AUC, F1, Precision, Recall with colour badges |

---

### 🔵 Original Baseline (original_baseline_backup)
```bash
python src/visualize-info/visualize_model.py --model_name densenet121 --checkpoint_path checkpoints/original_baseline_backup/best_model_auc.pth
```
*Saves to: `info/densenet121-test-output/`*

---

### 🟡 DenseNet-121 Optimized (densenet121_best_accuracy_run)
```bash
python src/visualize-info/visualize_model.py --model_name densenet121 --checkpoint_path checkpoints/densenet121_best_accuracy_run/best_model_auc.pth
```
*Saves to: `info/densenet121-test-output/`*

---

### 🏆 CheXNet Fine-tuned (chexnet_run) ← Best Model
```bash
python src/visualize-info/visualize_model.py --model_name chexnet --checkpoint_path checkpoints/chexnet_run/best_model_auc.pth
```
*Saves to: `info/CheXNet small-test-output/`*

---

### Future Models (run after training)
```bash
# EfficientNet-B4
python src/visualize-info/visualize_model.py --model_name efficientnet_b4 --checkpoint_path checkpoints/effnet_run/best_model_auc.pth

# Swin-T
python src/visualize-info/visualize_model.py --model_name swin_t --checkpoint_path checkpoints/swin_run/best_model_auc.pth

# ResNet-50
python src/visualize-info/visualize_model.py --model_name resnet50 --checkpoint_path checkpoints/resnet50_run/best_model_auc.pth
```

