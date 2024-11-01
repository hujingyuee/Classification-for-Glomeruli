import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from torchvision.models import resnet18

# Load the trained model
def load_model(model_path):
    model = resnet18()  # Instantiate the model class
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Modify the final layer to match trained model's output size
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

# Define transformations to match those used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to the input size expected by the model
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Normalization (if used during training)
])

# Function to predict class of an image
def predict(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def evaluate(folder_path, model_path, output_csv='evaluation.csv'):
    model = load_model(model_path)
    results = []

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if image_path.endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image formats
            prediction = predict(model, image_path)
            results.append([image_name, prediction])

    # Save results to a CSV file
    df = pd.DataFrame(results, columns=['filename', 'predicted_class'])
    df.to_csv(output_csv, index=False)
    print(f"Evaluation completed. Results saved to {output_csv}")

# Specify the paths directly in the script
if __name__ == "__main__":
    folder_path = 'E:/dataset/detect'
    model_path = 'E:/dataset/glomeruli_model.pth'
    evaluate(folder_path, model_path)
