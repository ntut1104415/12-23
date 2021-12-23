import streamlit as st
import cv2
import tensorflow as tf
import io
from PIL import Image
import tensorflow_addons as tfa
import numpy as np
from autocrop import Cropper


     Image = st.file_uploader('Upload your portrait here',type=['jpg','jpeg','png'])
        if Image is not None:
            col1, col2 = st.beta_columns(2)
            Image = Image.read()
            Image = tf.image.decode_image(Image, channels=3).numpy()                  
            Image = adjust_gamma(Image, gamma=gamma)
            with col1:
                st.image(Image)
            input_image = loadtest(Image,cropornot=Autocrop)
            prediction = comic_model(input_image, training=True)
            prediction = tf.squeeze(prediction,0)
            prediction = prediction* 0.5 + 0.5
            prediction = tf.image.resize(prediction, 
                           [outputsize, outputsize],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            prediction=  prediction.numpy()
            with col2:
                st.image(prediction)


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)

def loadtest(image,cropornot=False):
    if cropornot:
        Percent = st.sidebar.slider('Zoom adjust', min_value=50, max_value=100,value=50,step=5)
        cropper = Cropper(face_percent=Percent)

        # Get a Numpy array of the cropped image
        image_crop = cropper.crop(image)
        if image_crop is not None:
            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            image = tf.convert_to_tensor(image_crop, dtype=tf.float32)
        else:
            st.write('Cannot find your face to crop')
    
    image = (tf.cast(image, tf.float32) /255.0 *2) -1
    image = tf.image.resize(image, 
                           [256, 256],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.expand_dims(image, 0)
    return image

def loadframe(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = (tf.cast(image, tf.float32) /255.0 *2) -1
    image = tf.image.resize(image, 
                           [256, 256],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.expand_dims(image, 0)
    return image
