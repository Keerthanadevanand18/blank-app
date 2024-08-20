import streamlit as st
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

st.title("PyTorch and Streamlit Example")  
  
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])  
  
if uploaded_file is not None:  
   image = Image.open(uploaded_file)  
   st.image(image, caption="Uploaded Image", use_column_width=True)  
  
   # Preprocessing  
   transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])  
   img_tensor = transform(image).unsqueeze(0)  # Add batch dimension  
  
   # Load model  
   model = EfficientNetModel(num_classes=7)  
   output = model.predict(img_tensor)  
   if output is not None:  
      st.write(f"Model output: {output}")  
   else:  
      st.write("Model could not be loaded.")

