
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch
import torchvision.models as models

import torch
import torchvision.models as models

class EfficientNetModel:
    def __init__(self, num_classes):
        try:
            self.model = models.efficientnet_b0(pretrained=False)
            self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, num_classes)
            self.model.load_state_dict(torch.load('efficientnet_model_state_dict.pth'))
        except FileNotFoundError:
            print("Model file not found. Please check the path.")
        except RuntimeError as e:
            print(f"Error loading model state dict: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def predict(self, image_tensor):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
        return outputs


# Instantiate the model and load the state dictionary
model = EfficientNetModel(num_classes=7)
model.load_state_dict(torch.load('efficientnet_model_state_dict.pth'))
model.eval()  # Set the model to evaluation mode

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust based on your model's requirements
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Mapping from numeric indices to dx values
idx_to_dx = {
    0: 'Melanoma',   
    1: 'Basal Cell Carcinoma',   
    2: 'Actinic Keratosis', 
    3: 'Vascular Lesions',  
    4: 'Dermatofibroma',    
    5: 'Nevus',    
    6: 'Squamous Cell Carcinoma'    
}

# Clinical suggestions for specific diagnoses
suggestions = {
    'Melanoma': "Melanoma is a serious form of skin cancer. If you suspect melanoma, please seek immediate medical consultation. Early detection and treatment are crucial.",
    'Basal Cell Carcinoma': "Basal Cell Carcinoma (BCC) is a common skin cancer that is generally less aggressive. However, it is still important to have it evaluated and treated by a healthcare professional.",
    
}

# Streamlit UI
st.title('Skin Lesion Classification')
st.write("Upload an image of a skin lesion and the model will predict the type of lesion.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        prediction_idx = torch.argmax(output, dim=1).item()
        prediction_label = idx_to_dx[prediction_idx]

    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write(f'Prediction: {prediction_label}')
    
    # Display clinical suggestion
    if prediction_label in suggestions:
        st.write(suggestions[prediction_label])

