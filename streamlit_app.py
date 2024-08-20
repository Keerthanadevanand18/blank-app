import streamlit as st
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

@st.cache  
def load_model(num_classes):  
 try:  
      model = models.efficientnet_b0(pretrained=True)  
      model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)  
      # Uncomment if loading custom weights  
      # model.load_state_dict(torch.load('efficientnet_model_state_dict.pth'), strict=False)  
      return model  
 except Exception as e:  
      st.error(f"Error loading model: {e}")  
      return None

class EfficientNetModel:  
   def __init__(self, num_classes):  
      self.model = load_model(num_classes)  
  
   def predict(self, image_tensor):  
      if self.model:  
        self.model.eval()  
        with torch.no_grad():  
           outputs = self.model(image_tensor)  
        return outputs  
      else:  
        st.error("Model is not loaded.")  
        return None

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

