
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
    if model.model:
        output = model.predict(img_tensor)
        if output is not None:
            st.write(f"Model output: {output}")
    else:
        st.write("Model could not be loaded.")
