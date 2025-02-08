import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import time
import streamlit as st

st.title('Skin Cancer Diagnosis Using Dermatoscopy Images')

network=load_model(r'C:\Users\Admin\OneDrive\Desktop\Streamlit(CNN)\skincancer.h5')

image_file=st.file_uploader("Upload Dermatoscopy Image of Skin Lesion to Classify",type=['jpg','jpeg','png'])

if image_file:
    st.toast("File Uploaded Succesfully !")
    uploaded_image=Image.open(image_file)
    uploaded_image=uploaded_image.resize((224,224))
    st.image(uploaded_image,caption="Uploaded Image")
    def process_image(image):
        image=tf.image.resize(image,(224,224))
        image_array=np.array(image)
        if len(image_array.shape)==2:
            image_array=np.stack((image_array,)*3,axis=-1)
        image_array/=255.0

        image_array=np.expand_dims(image_array,axis=0)

        return image_array
    if st.button("Classify"):
        with st.spinner("Processing..."):
           time.sleep(6)
        image_arr=process_image(uploaded_image)
        predictions=network.predict(image_arr)
        classes=['Benign','Malignant']
        predicted_class=classes[np.argmax(predictions)]
        prob=np.max(predictions)
        if(predicted_class==classes[0]):
          st.success(f"The skin lesion is {predicted_class} with a Chance of {prob * 100:.2f}%")
        elif(predicted_class==classes[1]):
          st.error(f"The skin lesion is {predicted_class} with a Chance of {prob * 100:.2f}%")
        st.info(f"Chance Of Benign - {(predictions[0][0])*100:.2f}%")
        st.info(f"Chance Of Malignant - {(predictions[0][1])*100:.2f}%")
