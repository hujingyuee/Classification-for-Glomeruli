# Glomeruli Classification Project

## Overview
This project involves the classification of glomeruli images into two categories: globally sclerotic and non-globally sclerotic. The model is built using a ResNet-18 architecture, fine-tuned on a custom dataset.

## Approach
The approach involves using a pre-trained ResNet-18 model from PyTorch's torchvision library. The model's final fully connected layer is replaced to accommodate binary classification. The training process includes data augmentation, model training, and evaluation using various performance metrics.

## Dataset Splitting
The dataset is split into training and testing sets using the `train_test_split` function. The details of the split are as follows:

- **Training Set**: Comprises 80% of the total dataset. It is used to train the model.
- **Testing Set**: Comprises 20% of the total dataset. It is used to evaluate the model's performance.

The split is performed with `random_state=42` to ensure reproducibility. The code for splitting the dataset is as follows:

## Machine Learning Pipeline
1. **Data Loading and Preprocessing**:
   - Images are loaded using a custom PyTorch Dataset class, `GlomeruliDataset`.
   - Data augmentation is applied using transformations such as random horizontal flip, random rotation, and color jitter.
   - Images are resized to 224x224 pixels and normalized.

2. **Model Architecture**:
   - A pre-trained ResNet-18 model is used.
   - The final fully connected layer is replaced with a linear layer with two output features for binary classification.

3. **Training**:
   - The model is trained using the Adam optimizer with a learning rate of 0.001.
   - A cosine annealing learning rate scheduler is used to adjust the learning rate during training.
   - The loss function used is CrossEntropyLoss.

4. **Evaluation**:
   - The model is evaluated using precision, recall, and F1 score.
   - Training and validation loss and accuracy are plotted for each epoch.

## Dataset
- The dataset is divided into training and validation sets using an 80-20 split.
- The dataset is expected to be in a CSV file format with image paths and labels.

## Performance Metrics
- **Precision**: Measures the accuracy of positive predictions.
- **Recall**: Measures the ability of the model to find all the relevant cases.
- **F1 Score**: Harmonic mean of precision and recall, providing a balance between the two.

## Results
- The model's performance is tracked over 20 epochs.
- Training and validation loss and accuracy are plotted to visualize the model's learning process.

## Dependencies
- Python 3.6
- PyTorch
- torchvision
- pandas
- scikit-learn
- matplotlib
- PIL (Pillow)

## How to Run
1. Ensure all dependencies are installed.
2. Place the dataset CSV and images in the specified directory.
3. Run the `train.py` script to train the model.
4. The trained model weights will be saved as `glomeruli_model.pth`.

## References
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)

## Acknowledgments
This project is inspired by various public repositories and publications in the field of medical image classification.
