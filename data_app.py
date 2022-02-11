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


#  loading model states
model_75x = CarRecognition75()
model_75x.load_state_dict(torch.load('model_state75.pt',
                                     map_location=device))
model_100x = CarRecognition100()
model_100x.load_state_dict(torch.load('model_state100.pt',
                                      map_location=device))

#  loading masks
mask_75 = torch.load('mask_75.pt', map_location=device)
mask_100 = torch.load('mask_100.pt', map_location=device)


#  instantiating ensemble
model_ex = EnsembleModels(model_75x, model_100x)


st.title(
    '''
    Car Classifier
    '''
)

st.info("""
###### *Usage Instructions:*
This app classifies car types as either Sedan, Coupe, Truck or SUV. For best results,
an isometric image of a car should be uploaded, that is,  one which shows both the front and 
side-view of an object; an example of this is provided in the sidebar. If this is not available, 
side-view images may be used.

Switch the model ensemble mode in case of wrong predictions as the other ensemble may be
better suited for that image instance.

###### *NOTE:*
The underlying model has not been trained with a semantic segmentation map and therefore is not able to 
detect the absence of a car in a given image. Only car images should be uploaded as uploading any other
image will result in a classification in the context of cars.

The model has also not be trained to recognise car rear-views, therefore rear view images should not be uploaded.
""")


image_file = st.file_uploader('Please upload image here:', type=['jpg', 'jpeg', 'png'])

st.sidebar.title('Typical Isometric Car Image')
st.sidebar.image('iso_image2.png')
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
    if mode_choice == 'average':
        return model_ex.average_confidence(img)
    else:
        return model_ex.priority(img)


def plot_shap(filepath, mask, size, model):
    """
    This function produces shap plots
    """
    image = cv2.imread(filepath)
    image = cv2.resize(image, (size, size))
    image = transforms.ToTensor()(image)

    explainer = shap.DeepExplainer(model, mask)
    shap_values = explainer.shap_values(image.view(-1, 3, size, size))

    shap_numpy = [np.swapaxes(np.swapaxes(x, 1, -1), 1, 2) for x in shap_values]
    test_numpy = np.swapaxes(np.swapaxes
                             (image.view(-1, 3, size, size).numpy(), 1, -1), 1, 2)

    fig = shap.image_plot(shap_numpy, test_numpy,
                          labels=['sedan', 'coupe', 'suv', 'truck'])
    plt.savefig('plot.png', dpi=200)
    pass


def output():
    try:
        save_img(image_file)
        st.image('image.jpg', width=365)
        st.success(classify_image('image.jpg'))
        response = st.selectbox('Would you like to know why the model has classified this image as such?', ['No', 'Yes'])
        if response == 'Yes':
            st.write('Just a minute...')
            plot_shap('image.jpg', mask_75, 75, model_75x)
            st.subheader('Explanation:')
            st.image('plot.png', width=750)
            st.info('''
            In the image above, the importance of significant pixels are color coded. The 
            presence of blue pixels reduce the likelihood of the image belonging to that 
            particular class while red pixels represent an increased likelihood. 
            
            The model predicts a particular class if there are more red pixels 
            or less blue pixels compared to other classes.  
            ''')
        else:
            st.write('All Done!')
    except AttributeError:
        st.write('Please upload image above')


output()

