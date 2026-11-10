import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from src.model import BrainTumorCNN, CLASS_NAMES
from src.gradcam import make_gradcam

st.set_page_config(page_title="Explainable Brain MRI Demo", page_icon="🧠", layout="wide")
st.title("🧠 Explainable AI for Brain MRI Classification")
st.caption("Educational mini-project — not a medical diagnostic tool.")

@st.cache_resource
def load_model():
    model = BrainTumorCNN(num_classes=len(CLASS_NAMES))
    state = torch.load("models/brain_tumor_cnn.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

uploaded = st.file_uploader("Upload a brain MRI image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    tensor = transform(image).unsqueeze(0)

    left, right = st.columns(2)
    with left:
        st.subheader("Uploaded MRI")
        st.image(image, use_container_width=True)

    try:
        model = load_model()
        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
        index = int(torch.argmax(probabilities))
        label = CLASS_NAMES[index]
        confidence = float(probabilities[index]) * 100

        with right:
            st.subheader("Model output")
            st.metric("Predicted class", label)
            st.metric("Model confidence", f"{confidence:.1f}%")
            st.write("Class probabilities")
            for name, prob in zip(CLASS_NAMES, probabilities.tolist()):
                st.progress(prob, text=f"{name}: {prob*100:.1f}%")

        st.subheader("Grad-CAM explanation")
        heatmap = make_gradcam(model, tensor, target_class=index)
        image_np = np.array(image.resize((224, 224)))
        fig, ax = plt.subplots()
        ax.imshow(image_np)
        ax.imshow(heatmap, cmap="jet", alpha=0.38)
        ax.axis("off")
        st.pyplot(fig)
        st.info("The heatmap highlights image regions that influenced this model's prediction. It is an explanation aid, not proof of tumor location.")
    except FileNotFoundError:
        st.error("Trained model not found. First run: python train.py")
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
else:
    st.info("Upload an MRI image to see a prediction and Grad-CAM visualization.")

st.warning("This is a student demonstration. It has not been clinically validated and must not be used for diagnosis or treatment decisions.")
