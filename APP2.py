import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Define the ConvNet model (same as before)
class ConvNet(nn.Module):
    """CNN Model for Bone Fracture Binary Classification."""
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 25 * 25, 128),  # Adjusted based on the actual feature map size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load the trained model and map weights to CPU if necessary
def load_model(model_path):
    model = ConvNet(num_classes=2)
    # Load the model weights and map them to the CPU if CUDA is not available
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Define transformations for input image
def transform_image(image):
    data_transform = transforms.Compose([
        transforms.Resize((100, 100)),  # Resize images to a consistent size
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale images to RGB
        transforms.ToTensor(),           # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
    ])
    return data_transform(image).unsqueeze(0)  # Add batch dimension

# Define the Streamlit interface
def main():
    st.title("Bone Fracture Detection")
    
    st.header("Upload an Image of a Bone to Classify if it is Fractured or Not")
    
    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        # Open the image and display it
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Transform image for prediction
        image_tensor = transform_image(image)
        
        # Load model
        model = load_model('bone_fracture_detection_model_weights.pth')
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
        
        # Display prediction result with swapped labels
        if predicted.item() == 1:  # Swapped labels: 1 -> Not Fractured, 0 -> Fractured
            st.write("The image is classified as **Fractured**.")
        else:
            st.write("The image is classified as **Not Fractured**.")
        
    # Additional functionality for visualization (optional)
    st.sidebar.header("Training Loss and Accuracy")
    # Add the option to display the plot from training
    if st.sidebar.button('Show Training and Validation Loss/Accuracy Plot'):
        img_path = 'train_val_loss_acc_plot.png'  # Ensure this is saved before running Streamlit
        st.image(img_path, caption="Training and Validation Loss/Accuracy Plot", use_column_width=True)

# Run the Streamlit app
if __name__ == '__main__':
    main()
