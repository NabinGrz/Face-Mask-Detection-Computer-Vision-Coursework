import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Face Mask Detection System",
    layout="centered"
)

st.title("Face Mask Detection System")
st.write("CNN-based Image Classification using Transfer Learning")

# --------------------------------------------------
# Device Configuration
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Running on: **{device}**")

# --------------------------------------------------
# File Check (CRITICAL FOR STREAMLIT)
# --------------------------------------------------
if not os.path.exists("model.pth"):
    st.error("❌ model.pth not found in project directory.")
    st.stop()

# --------------------------------------------------
# Load Model Safely
# --------------------------------------------------
try:
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(1280, 2)

    checkpoint = torch.load("model.pth", map_location=device)

    # Support both save formats (safe)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        class_names = checkpoint.get(
            "class_names", ["with_mask", "without_mask"]
        )
    else:
        model.load_state_dict(checkpoint)
        class_names = ["with_mask", "without_mask"]

    model.to(device)
    model.eval()

    st.success("✅ Model loaded successfully")

except Exception as e:
    st.error("❌ Failed to load model")
    st.exception(e)
    st.stop()

# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# Image Upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, 1).item()

    label = class_names[pred_idx]
    confidence = probs[0][pred_idx].item()

    st.markdown("### 🔍 Prediction Result")
    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence:** {confidence * 100:.2f}%")

else:
    st.info("👆 Please upload an image to get a prediction.")
