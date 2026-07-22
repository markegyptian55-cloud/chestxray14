import os
import time
import re

report_path = r"D:\project\DEEP LEARN PROJECT\NIH Chest X-rays\Dataset\info\CheXNet small-test-output\evaluation_report.txt"
book_path = r"D:\project\DEEP LEARN PROJECT\NIH Chest X-rays\Dataset\info\book.md"

print("Starting async book updater daemon...")

for i in range(180): # Poll every 10 seconds for up to 30 minutes
    if os.path.exists(report_path):
        print("Found evaluation report! Waiting 2s for file to finish writing...")
        time.sleep(2)
        
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse Class AUCs
            auc_matches = re.findall(r"^([\w_]+)\s*\|\s*([\d\.]+)", content, re.MULTILINE)
            auc_dict = {disease: score for disease, score in auc_matches if disease != "Mean"}
            
            # Parse Mean AUC
            mean_match = re.search(r"Mean AUC-ROC\s*\|\s*([\d\.]+)", content)
            mean_auc = mean_match.group(1) if mean_match else "N/A"
            
            # Parse classification report table
            clf_report = ""
            report_start = content.find("Classification Metrics (Threshold = 0.5):")
            if report_start != -1:
                clf_report = content[report_start:]
                
            # Read book.md
            with open(book_path, 'r', encoding='utf-8') as f:
                book_content = f.read()
                
            # Build CheXNet Section
            chexnet_section = f"""
---

## Chapter 13: CheXNet Final Test Set Evaluation & Comparison

This chapter documents the final evaluation metrics for the **CheXNet** model (pre-trained on NIH Chest X-rays) on the independent test set (25,596 images), and compares it directly with the ImageNet-initialized DenseNet-121 model.

### 1. CheXNet Test Set AUC-ROC Scores
* **Model Checkpoint:** `checkpoints/chexnet_run/best_model_auc.pth` (Epoch 5 Weights)
* **Mean AUC-ROC:** **{mean_auc}**

| Pathology / Disease | CheXNet AUC | DenseNet-121 AUC | Difference |
| :--- | :---: | :---: | :---: |
"""
            
            # DenseNet-121 AUC scores from book.md for comparison
            densenet_aucs = {
                "Atelectasis": "0.7798", "Cardiomegaly": "0.8759", "Effusion": "0.8327",
                "Infiltration": "0.7004", "Mass": "0.8249", "Nodule": "0.8003",
                "Pneumonia": "0.7325", "Pneumothorax": "0.8600", "Consolidation": "0.7463",
                "Edema": "0.8440", "Emphysema": "0.9212", "Fibrosis": "0.8433",
                "Pleural_Thickening": "0.7966", "Hernia": "0.9240"
            }
            
            for disease, score in auc_dict.items():
                std_name = disease.replace(" ", "_")
                dn_score = densenet_aucs.get(std_name, "N/A")
                try:
                    diff = float(score) - float(dn_score)
                    diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
                except:
                    diff_str = "N/A"
                chexnet_section += f"| **{disease}** | `{score}` | `{dn_score}` | **{diff_str}** |\n"
                
            try:
                diff_mean = float(mean_auc) - 0.8201
                diff_mean_str = f"+{diff_mean:.4f}" if diff_mean >= 0 else f"{diff_mean:.4f}"
            except:
                diff_mean_str = "N/A"
                
            chexnet_section += f"| **Mean AUC-ROC** | **`{mean_auc}`** | **`0.8201`** | **{diff_mean_str}** |\n"
                
            chexnet_section += f"""
### 2. CheXNet Classification Report (Threshold = 0.5)

```text
{clf_report}
```
"""
            # Append to book.md if not already appended
            if "Chapter 13: CheXNet Final Test Set Evaluation" not in book_content:
                # Update Table of Contents
                toc_target = "* [Chapter 12: Troubleshooting & CheXNet Setup Details](#chapter-12-troubleshooting--chexnet-setup-details)"
                toc_replacement = toc_target + "\n* [Chapter 13: CheXNet Final Test Set Evaluation & Comparison](#chapter-13-chexnet-final-test-set-evaluation--comparison)"
                book_content = book_content.replace(toc_target, toc_replacement)
                
                book_content += chexnet_section
                with open(book_path, 'w', encoding='utf-8') as f:
                    f.write(book_content)
                print("Successfully updated book.md with CheXNet evaluation comparison!")
            else:
                print("Book already updated. Exiting.")
            break
        except Exception as e:
            print(f"Error updating book: {str(e)}")
            break
            
    time.sleep(10)
