import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import json

# Title of the app
st.title("Image Classification with ResNet18")

# Sidebar for model information
st.sidebar.header("Model Information")
st.sidebar.write("This model is ResNet18 trained to classify images as hotdog or not hotdog.")

# Load the model from Hugging Face
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="asidfactory/hotdognothotdog", filename="resnet_state_dict.pth")
    config_path = hf_hub_download(repo_id="asidfactory/hotdognothotdog", filename="config.json")
    
    # Load the model configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    # Define the model architecture
    class ResNet18(torch.nn.Module):
        def __init__(self, num_classes):
            super(ResNet18, self).__init__()
            self.resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
            self.resnet18.fc = torch.nn.Linear(self.resnet18.fc.in_features, num_classes)

        def forward(self, x):
            return self.resnet18(x)
    
    # Initialize and load weights
    model = ResNet18(num_classes=config["num_classes"])
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Set to evaluation mode
    return model, config["classes"], config["normalize_mean"], config["normalize_std"]

model, classes, normalize_mean, normalize_std = load_model()

# Image preprocessing function
def preprocess_image(image, input_size, normalize_mean, normalize_std):
    transform = transforms.Compose([
        transforms.Resize((input_size[1], input_size[2])),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_tensor = preprocess_image(image, [3, 256, 256], normalize_mean, normalize_std)

    # Make a prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = classes[predicted.item()]

    # Display prediction
    st.write(f"Predicted class: **{predicted_class}**")

