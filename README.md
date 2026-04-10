# sign-language-image-recognition
A convolutional image classifier that recognises the first three letters of German Sign Language (A, B, C), built with TensorFlow and transfer learning (EfficientNetV2B0). Trained on a collaboratively collected real-world dataset with group-aware splitting. Achieved 98.49% vs. 95.45% accuracy on an internal test-data vs. a held-out test set from the bootcamp challenge.


## Project Overview

The goal was to build an end-to-end image classification pipeline - from data collection through training and evaluation. This included experimenting with custom CNN architectures as well as transfer learning approaches, ultimately landing on a fine-tuned pre-trained backbone as the most effective solution. The dataset was collected collaboratively: each participant in the bootcamp cohort, and also several instructors photographed their own hands forming the three signs, pooling images together to get a more diverse training set.

What made this more than a toy problem:

- The data was collected under real conditions (varied lighting, backgrounds, hand shapes), not scraped from the web.
- A **group-aware train/val/test split** was used to ensure the model was evaluated on hands it had never seen during training - a stricter test of generalisation than a random split would provide.
- The project progressed through several stages of increasing complexity, which is documented in the notebook.


## Results

| **Metric**    | **Model training, validation & test data set**             | **Challenge test set**  |
| ------------- | ---------------------------------------------------- | ----------------------- |
| Test Accuracy | **98.49%**                                           | **95.45%**              |
| Test Images   | 1,379 (981 / 199 / 199)                                                | 1,188                   |
| Model         | EfficientNetV2B0 (fine-tuned)                                                  |


## Confusion Matrix

The fine-tuned model achieves 98.49% accuracy on the internal test set (196/199 correct), with only three misclassifications: two A samples predicted as B and one C predicted as A. On the challenge set the overall accuracy drops slightly to 95.45% (1,134/1,188). The gap is driven almost entirely by class A, which falls from 95.3% (41/43) internally to 89.1% (353/396) on the challenge set, with 39 of those errors going to class C. Classes B and C are robust across both evaluations — B scores 100% internally and 97.7% on the challenge set, while C scores 98.9% and 99.5% respectively. The A→C confusion on the challenge set likely reflects hand angles or framing conditions in that dataset where the two signs look visually similar — an edge case the internal test set happens to contain very little of.


Model test set             |  Challenge test set
:-------------------------:|:-------------------------:
<img width="640" height="545" alt="confusion_matrix_training" src="https://github.com/user-attachments/assets/ff066f8f-928b-4fdb-9e42-d47339b028a1" />  |  <img width="551" height="906" alt="Screenshot 2026-04-09 at 13 09 59" src="https://github.com/user-attachments/assets/2649deda-196e-475c-80fe-681805217db6" />


## Data

- ~60 photos per participant per class, across two collection rounds (plain backgrounds, then varied backgrounds)
- Images resized to 256×256 and padded to maintain aspect ratio
- Group-aware split: no participant's images appear in more than one split

## Model Architecture

The model architecture was not fixed from the start - I experimented with several backbones (including custom CNN architectures and other EfficientNet variants) before settling on EfficientNetV2B0, which gave the best balance of accuracy and training stability on this dataset.

The final pipeline consists of four sequential components:

Input (256×256×3)

→ Data Augmentation (random flip, rotation, contrast)

→ EfficientNetV2 Preprocessor

→ EfficientNetV2B0 Backbone (frozen → then fine-tuned)

→ Global Average Pooling

→ Dense(64, ReLU)

→ Dense(3, Softmax)

## Training Strategy

- **Warm-up phase:** Backbone frozen; only the classification head trained. This prevents the untrained head's large gradients from disrupting the pre-trained backbone weights.
- **Fine-tuning phase:** Backbone unfrozen; full model trained at a lower learning rate to specialise the features for sign language.

## Regularisation

- **Image augmentation:** random horizontal flips, rotations (±90°), contrast adjustments
- **AdamW optimiser** with weight decay (L2 regularisation)
- **EarlyStopping** on validation loss with restore_best_weights=True
- **ModelCheckpoint** to preserve the best epoch during fine-tuning

## Repository Structure

sign-language-classifier/
```
├── notebooks/

│ └── transfer_learning_sign_language.ipynb # Full training pipeline

├── data/ # not included (see note below)

├── results/

│ └── confusion_matrix_results.png # Final evaluation screenshot

├── requirements.txt

└── README.md
```

**Note on data:** The dataset consists of photos taken by bootcamp participants and is not publicly redistributable. To evaluate whether the model generalises to new people rather than just new photos of the same people, a group-aware train/val/test split is used: each participant is treated as a group and kept entirely within one split. A Monte Carlo approach finds the best assignment of whole groups to each split while approximating the target size ratios (87.5% / 12.5% / 12.5%). The test seed is fixed to ensure a consistent held-out set; the validation seed can be varied to check that results are stable. To reproduce results, you would need to collect your own images of A/B/C hand signs, organise them into subdirectories, and run the preprocessing steps as described in the notebook.

## Setup
```
\# Clone the repository

git clone <https://github.com/<your-username>/sign-language-classifier.git>

cd sign-language-classifier

\# Install dependencies

pip install -r requirements.txt

\# Launch notebook

jupyter notebook notebooks/transfer_learning_sign_language.ipynb
```

## Requirements

See requirements.txt. Key dependencies:

- TensorFlow / Keras
- scikit-learn
- NumPy, Matplotlib, Seaborn

**Apple Silicon (M1/M2/M3/M4):** The standard tensorflow package in requirements.txt targets Linux/Windows. On macOS with Apple Silicon, replace it with the Metal-accelerated build:

```
pip install tensorflow-macos==2.16.2 tensorflow-metal==1.2.0
```

## My key learnings

This project was my first time working with real image data collected in the wild rather than a clean benchmark dataset. A few things stood out:

- **Group-aware splitting matters.** A random split would have inflated the test accuracy significantly, since a model can generalise to new photos of the same hands much more easily than to entirely new hands.
- **Transfer learning is remarkably effective even on small datasets.** The EfficientNetV2B0 backbone, pre-trained on ImageNet, provided strong enough feature representations that the classification head needed very little data to learn the task.
- **Fine-tuning requires patience.** Training loss spikes noticeably when the backbone is unfrozen - partly due to the backbone's internal regularisation mechanisms - but validation loss continues to improve, which is the metric that actually matters.
- **Data diversity beats data quantity.** Adding images with varied backgrounds in the second collection round improved generalisation more than simply adding more plain-background photos would have.

## Context

Built during the **Data Science & AI Bootcamp** as a hands-on applied project covering the full computer vision pipeline: from pixel to prediction.
