# Satellite Image Classification with a CNN (EuroSAT)

A convolutional neural network, written and trained from scratch in PyTorch (no pretrained weights), that classifies 64×64 satellite images into 10 land-use classes. The repo also contains a small web app to try the model on your own images.

| | |
|---|---|
| **Validation accuracy** | 97.6 % (best epoch 82 of 120) |
| **Test accuracy** | 97 % (scored on the course challenge server) |
| **Parameters** | about 1.7 M (6.9 MB weights file) |
| **Data** | 10,000 images, 90/10 stratified train/validation split, seed 42 |

The model was also the most compact and fastest to deploy among about 500 submissions in the course challenge.

## Task

Land-use classification on a 10,000-image subset of EuroSat (Sentinel-2 satellite images, 64×64 RGB), provided for the course *Programming in Python II* at JKU Linz. The ten classes are `AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River` and `SeaLake`.

![Random samples](docs/random_samples.svg)

## Preprocessing
- Split: 90/10, stratified by class, fixed seed, so results are reproducible.
- Normalization: per-channel mean and standard deviation computed on the training split only, to avoid leakage into validation and test.
- Augmentation: random flips and rotation, training set only. Satellite images have no fixed orientation, so these are label-preserving.
- Consistency: validation, test and the web app use the same deterministic transform (resize, tensor conversion, normalization).
- EDA: pixel and brightness distributions per class were checked before training. Color transformations of images would cause a distortion of important features.

## Model

Five convolutional layers, each followed by batch normalization and ReLU, with three max-pooling steps, global average pooling and a small classifier head:

```
Conv(3→64) → Conv(64→64) → MaxPool → Conv(64→128) → MaxPool →
Conv(128→256) → MaxPool → Conv(256→512) → GlobalAvgPool →
Linear(512→256) → ReLU → Dropout(0.5) → Linear(256→10)
```

The definition is in [`model.py`](model.py).

## Training

- Loss and optimizer: cross-entropy, Adam with learning rate 1e-3
- Learning-rate schedule: `StepLR`, halved every 15 epochs
- Batch size 64, 120 epochs, the checkpoint with the best validation accuracy is kept
- Augmentation: random horizontal and vertical flips, random rotation up to 90°
- Input normalization: per-channel mean and standard deviation, computed on the training split only to avoid data leakage

![Training curves](docs/model_performance.svg)

## Evaluation

![Confusion matrix](docs/confusion_matrix.svg)

Examples of validation images the model got wrong:

![Misclassified samples](docs/misclassified_samples.svg)

Further data exploration (pixel value distribution, brightness per class) is in the notebook.

## Repository

```
eurosat_cnn.ipynb      data handling, EDA, model, training, evaluation, test predictions
model.py               model definition, label map, normalization constants
app.py                 Shiny for Python web app for inference
assets/weights/        best_model.pth (trained weights)
docs/                  figures used in this README
```

## Run it

```bash
pip install -r requirements.txt

# web app (weights are included, no data needed)
shiny run app.py
```

To re-run the notebook, place the data next to it as `data/<ClassName>/*.jpg` (labelled images) and `public_test_data/*.jpg` (unlabelled test images). The data is not part of this repository.

The web app takes an uploaded image, resizes it to 64×64, normalizes it with the training statistics and shows the predicted class together with all ten class probabilities.
