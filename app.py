import streamlit as st
import streamlit.components.v1 as stc
from os import walk, path
import os
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
# import cv2
from PIL import Image


def load_image(image_file):
    img = Image.open(image_file).convert("RGB")
    im1 = img.save("temp.jpg")
    img = tf.io.read_file("./temp.jpg")
    os.remove("./temp.jpg") 
    return img

def preprocess_image(img):
    img = tf.image.decode_image(img, channels=3)        # making sure image has 3 channels
    img = tf.image.convert_image_dtype(img, tf.float32) # making sure image has dtype float 32
    img = img[tf.newaxis, :]
    return img

def main():
    st.title("Neural Style Transfer")
    
    st.write("### Select only two images")

    st.write("#### First select the :blue[content image] and then select the :blue[style image.]")

    images = st.file_uploader("Upload Image", type=['png', 'jpeg', 'jpg'], accept_multiple_files=True)
    
    st.image(images,width=650)

    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    if st.button("Process"):
        if images is not None:
            content_image = load_image(images[0])
            style_image = load_image(images[1])
            content_image = preprocess_image(content_image)
            style_image = preprocess_image(style_image)
            style_image = tf.image.resize(style_image, (256, 256))
            stylized_image = np.array(style_image)

            stylized_image = stylized_image.reshape(stylized_image.shape[1],stylized_image.shape[2],stylized_image.shape[3])
            stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
            
            st.write('### ----------------------------------------------------------------------')
            st.write('### :green[Stylized Image]')
            st.image((np.squeeze(stylized_image)),width=650)

if __name__ == '__main__':
    main()