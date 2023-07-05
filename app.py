import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model('model.h5')
# Define the image size for model input
IMG_SIZE = (128, 128)

# Set the app title and sidebar
# Add custom CSS for aesthetics
st.markdown(
    """
    <style>
    .title {
        margin-top:0px;
        color: #FF5733; /* Coral */
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .text {
        color: #EFA18A; /* Slate Gray */
        font-size: 20px;
        font-weight: italic;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .uploaded-image {
        width: 100%;
        max-width: 500px;
        margin-bottom: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .prediction {
        color: #FF5733; /* Coral */
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .confidence {
        color: #FF5600; /* Coral */
        font-size: 18px;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_option('deprecation.showPyplotGlobalUse', False)

# Display the title
st.markdown("<h1 class='title'>Alzheimer's Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h1 class='text'>Alzheimer's Disease Prediction is a web application that utilizes a pre-trained deep learning model to predict the presence of Alzheimer's disease based on uploaded brain ultrasound images. Users can upload an image through the sidebar and the app will process the image using the trained model.</h1>", unsafe_allow_html=True)

st.sidebar.title("Upload Image")
st.sidebar.markdown("Please upload an image.")


def preprocess_image(image):
    # plt.imsave('image2.jpg', image)
    img_array = np.array(image)
    rgb_image = np.repeat(img_array[:, :, np.newaxis], 3, axis=2)
    img = Image.fromarray(img_array.astype('uint8'))


    # img.save('output1.jpg')  # Save the image to a file

    img_array = np.expand_dims(rgb_image, axis=0)
    return img_array





def predict(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    # print(prediction)
    predicted_idx = np.argmax(prediction, axis=1)[0]
    return predicted_idx

# Display the file uploader
uploaded_file = st.sidebar.file_uploader(label="", type=['jpg', 'jpeg', 'png'])

# Make predictions and display the result
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    predicted_idx = predict(image)
    
    class_labels = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
    predicted_label = class_labels[predicted_idx]

    st.markdown(f"<p class='prediction'>Prediction: {predicted_label}</p>", unsafe_allow_html=True)

else:
    st.sidebar.write("Please upload an image.")