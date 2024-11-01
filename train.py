import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, ColorJitter, ToTensor, Normalize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet18
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

class GlomeruliDataset(Dataset):
    def __init__(self, annotations, root_dir, transform=None):
        self.annotations = annotations
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        img_name = self.annotations.iloc[index, 0]
        label = int(self.annotations.iloc[index, 1])


        if label == 1:  # globally sclerotic
            img_path = os.path.join(self.root_dir, 'globally_sclerotic_glomeruli', img_name)
        else:  # non-globally sclerotic
            img_path = os.path.join(self.root_dir, 'non_globally_sclerotic_glomeruli', img_name)


        image = Image.open(img_path).convert("RGB")
        image = image.resize((224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label

train_transform = Compose([
    RandomHorizontalFlip(),
    RandomRotation(15),
    ColorJitter(brightness=0.2, contrast=0.2),
    ToTensor(),
    Normalize((0.5,), (0.5,))
])

test_transform = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))
])

csv_file_path = 'E:/dataset/public.csv'
root_dir_path = 'E:/dataset/photo'

annotations = pd.read_csv(csv_file_path)
train_annotations, test_annotations = train_test_split(annotations, test_size=0.2, random_state=42)

# load the dataset
train_dataset = GlomeruliDataset(annotations=train_annotations, root_dir=root_dir_path, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = GlomeruliDataset(annotations=test_annotations, root_dir=root_dir_path, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# model setting
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

# train loop
num_epochs = 20
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
val_precisions = []
val_recalls = []
val_f1s = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_epoch_loss = val_running_loss / len(test_loader)
    val_epoch_accuracy = val_correct / val_total
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_accuracy)

    # Calculate precision, recall, and F1 score
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    val_precisions.append(precision)
    val_recalls.append(recall)
    val_f1s.append(f1)

    print(f"Epoch {epoch + 1}/{num_epochs}, "
          f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, "
          f"Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.4f}, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.show()

# save the model
torch.save(model.state_dict(), "glomeruli_model.pth")
