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

Epoch 1/10, Train Loss: 0.1629, Train Accuracy: 0.9427, Val Loss: 0.1388, Val Accuracy: 0.9410, Precision: 1.0000, Recall: 0.6583, F1 Score: 0.7939

Epoch 2/10, Train Loss: 0.0873, Train Accuracy: 0.9677, Val Loss: 0.0557, Val Accuracy: 0.9757, Precision: 0.9524, Recall: 0.9045, F1 Score: 0.9278

Epoch 3/10, Train Loss: 0.0722, Train Accuracy: 0.9768, Val Loss: 0.0439, Val Accuracy: 0.9818, Precision: 0.9198, Recall: 0.9799, F1 Score: 0.9489

Epoch 4/10, Train Loss: 0.0659, Train Accuracy: 0.9770, Val Loss: 0.0610, Val Accuracy: 0.9809, Precision: 0.9538, Recall: 0.9347, F1 Score: 0.9442

Epoch 5/10, Train Loss: 0.0582, Train Accuracy: 0.9785, Val Loss: 0.0925, Val Accuracy: 0.9774, Precision: 0.9832, Recall: 0.8844, F1 Score: 0.9312

Epoch 6/10, Train Loss: 0.0624, Train Accuracy: 0.9805, Val Loss: 0.0518, Val Accuracy: 0.9870, Precision: 0.9646, Recall: 0.9598, F1 Score: 0.9622

Epoch 7/10, Train Loss: 0.0367, Train Accuracy: 0.9863, Val Loss: 0.0451, Val Accuracy: 0.9818, Precision: 0.9363, Recall: 0.9598, F1 Score: 0.9479

Epoch 8/10, Train Loss: 0.0450, Train Accuracy: 0.9846, Val Loss: 0.0340, Val Accuracy: 0.9887, Precision: 0.9604, Recall: 0.9749, F1 Score: 0.9676

Epoch 9/10, Train Loss: 0.0367, Train Accuracy: 0.9857, Val Loss: 0.0635, Val Accuracy: 0.9783, Precision: 0.9028, Recall: 0.9799, F1 Score: 0.9398

Epoch 10/10, Train Loss: 0.0310, Train Accuracy: 0.9891, Val Loss: 0.0331, Val Accuracy: 0.9913, Precision: 0.9896, Recall: 0.9598, F1 Score: 0.9745

![553eef3072362c1032c2f536b15719ce](https://github.com/user-attachments/assets/ad40090a-8809-4dd8-a781-3fdc9e1ac14c)
![ac69162a3aa4f4a37a6171d50d14f124](https://github.com/user-attachments/assets/2837664c-e6ae-438c-85c0-9a1fb4ed16fe)


