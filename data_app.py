import streamlit as st
import numpy as np
import shap
import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import shutil
import os
from helper import EnsembleModels, CarRecognition75, CarRecognition100

#  configuring device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def load_models():
    model_75x = CarRecognition75()
    model_75x.load_state_dict(torch.load('model_states/model_state75.pt',
                                         map_location=device))
    model_100x = CarRecognition100()
    model_100x.load_state_dict(torch.load('model_states/model_state100.pt',
                                          map_location=device))

    #  instantiating ensemble
    model_ex = EnsembleModels(model_75x, model_100x)
    return model_ex


st.title(
    '''
    Car Classifier
    '''
)

st.info("""
###### *Usage Instructions:*
This app classifies cars as either Sedan, Coupe, Truck or SUV. For best results,
an isometric image of a car should be uploaded, that is, one which shows both the front and 
side-view of the car, with the car occupying most of the frame; an example of this 
is provided in the sidebar (if not visible, tap the arrow at the top edge). However, if this 
is not available side-view images may be used.

Feel free to switch model ensemble modes in case of wrong predictions as the other mode may be
better suited for that particular image instance.

###### *NOTE:*
The underlying model has not been trained with a semantic segmentation map and is therefore not able to 
detect the absence of a car in a given image. Only car images should be uploaded as uploading any other
image will result in image classification in the context of the predefined car classes.

The model has also not be trained to recognise car front or rear-views, therefore such images should not
be uploaded as they will yield inaccurate results.
""")

st.write('Do you need some description of the car classes?')
description = st.checkbox('Show description')
if description:
    st.image('images/sedan.png', width=300)
    st.info(
        """
        ###### *Sedan:*
        A sedan is a passenger car with a three-box chassis configuration where the engine compartment, passenger cabin
        and cargo compartment are easily distinguishable. Sedans are characterised by their 4-door design with two rows
        of seats and a subtle curvature of the chassis roofline and rear.
        """
    )

    st.image('images/coupe.png', width=300)
    st.info(
        """
        ###### *Coupe:*
        A Coupe is a car similar in design to a Sedan but with differences such as a 2-door design, prominently curved
        roofline and rear, reduced headroom in the passenger cabin, lower ground clearance and an aerodynamic chassis.
        """
    )

    st.image('images/suv.png', width=300)
    st.info(
        """
        ###### *SUV*
        A Sports Utility Vehicle or SUV is a car characterised by its bold chassis design, high ground clearance,
        significant headroom in the passenger cabin and a cargo compartment which is an extension of the passenger cabin.
        """
    )

    st.image('images/truck.png', width=300)
    st.info(
        """
        ###### *Truck*
        A Truck or Pickup Truck is a car designed to carry heavy cargo as evident by its pronounced cargo compartment.
        Just like an SUV, it also has a bold chassis design and high ground clearance making it suitable for off-road
        travel and haulage.
        """
    )


image_file = st.file_uploader('Please upload image here:', type=['jpg', 'jpeg', 'png'])

st.sidebar.title('Typical Isometric Car Image')
st.sidebar.image('images/iso_image2.png')
st.sidebar.subheader('Options')
mode_choice = st.sidebar.radio('Set Ensemble Mode', ['Average', 'Priority'])

st.text(
    '''
    Output:
    '''
)


def save_img(img):
    with open('image.jpg', 'wb') as f:
        f.write(img.getbuffer())
        pass


def classify_image(img):
    if mode_choice == 'Average':
        model_ex = load_models()
        return model_ex.average_confidence(img)
    else:
        model_ex = load_models()
        st.write('Priority Mode')
        return model_ex.priority(img)


def output():
    try:
        save_img(image_file)
        st.image('image.jpg', width=365)
        st.success(classify_image('image.jpg'))
        st.write('Done!')
        os.remove('image.jpg')
    except AttributeError:
        st.write('Please upload image above')


output()
