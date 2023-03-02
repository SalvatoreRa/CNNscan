#structural libraries
import streamlit as st
import io
from PIL import Image, ImageFilter
from io import BytesIO
import requests
import os
import sys
import pathlib
import json
import urllib.request

#model specific libraries
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import matplotlib.pyplot as plt
import math
import torchvision.transforms as transforms
from torchvision import models
from sklearn.preprocessing import minmax_scale
from matplotlib import cm
from torch.nn import ReLU
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import SGD, Adam
import copy
from matplotlib.colors import ListedColormap
from copy import deepcopy
import networkx as nx
import random
from lime import lime_image
from skimage.segmentation import mark_boundaries
import pandas as pd

#We are importing from the modules
sys.path.append(str(pathlib.Path().absolute()).split("/src")[0] + "/src")
from utils import (load_test_image, load_baseline, 
    format_np_output, save_image, save_gradient_images, convert_to_grayscale, 
    process_img, save_class_activation_images,  
    apply_colormap_on_image, apply_heatmap, recreate_image, 
    preprocess_image, get_positive_negative_saliency, 
    guided_grad_cam, conv_layer_indices, read_imagenet_categ, 
    preprocess_and_blur_image, download_images
    )

from methods import ( fetch_filters, advance_filt, fetch_feature_maps, CamExtractor, GradCam, Visualize_GradCam,
    VanillaBackprop, VanillaBackprop_process, GuidedBackprop, GuidedBackprop_process, 
    scoreCamExtractor, ScoreCam, scorecam_process, GuidedGradCam, gradient_gradcam,
    IntegratedGradients, integrated_gradient_process, CNNLayerVisualization,
    LRP, LRP_process, LayerCam, LayerCAM_process, 
    Grad_times_process, generate_smooth_grad, smooth_grad_process,
    smooth_grad_process_guidBackprop, LR_GuidedBackprop, layer_act_guid_bp, InvertedRepresentation, 
    inverted_representation_process, ClassSpecificImageGeneration, class_generated_images, 
    DeepDream, dream, RegularizedClassSpecificImageGeneration, 
    regularized_class_img_gen, model_layers_to_df,
    hierarchy_pos, plot_conv_model_structure)

from outputs import cam_outputs, outputs_backprop, outputs_scorecam, \
    outputs_LRP, outputs_smoothgrad, output_adv_filt, output_layer_act_guid_bp, \
    outputs_DD, output_inverted, outputs_CGI

from description import (Aknowledgment, CNN_overview)


@st.cache(ttl=12*3600)
def load_model():
  model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
  }
  class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.gradients = None
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations(self, x):
        return self.features(x)
    
    def forward(self, x):
        x = self.features(x)

        hook = x.register_hook(self.activations_hook)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
  model = AlexNet()
  state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=True)
  model.load_state_dict(state_dict)
  return model

@st.cache(ttl=3600)
def VGG16():
    pret_mod =  models.vgg16(pretrained=True)
    return pret_mod

@st.cache(ttl=3600)
def VGG19():
    pret_mod =  models.vgg19(pretrained=True)
    return pret_mod

  



##########################################################################
########################  Main app               #########################
##########################################################################
# Create the main app
def main():
    model = load_model()

    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #000000;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">CNNscan</p>', unsafe_allow_html=True)
            
        
    with col2:
        img_logo = "https://github.com/SalvatoreRa/CNNscan/blob/main/img/logo.png?raw=true"
        response = requests.get(img_logo)
        logo = Image.open(BytesIO(response.content))               
        st.image(logo,  width=150)

    img_path = 'https://github.com/SalvatoreRa/CNNscan/blob/main/img/cnn_scan.png?raw=true'
    capt = 'An android holding a radiography of a robot. Image created by the author with DALL-E'
    response = requests.get(img_path)
    img_screen = Image.open(BytesIO(response.content))
    st.image(img_screen, caption=capt, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.subheader('Visualize what happening inside a convolutional neural network (CNN)')

    #### SIDEBAR #####
    st.sidebar.image(logo,  width=150)
    st.sidebar.markdown("Made by [Salvatore Raieli](https://www.linkedin.com/in/salvatore-raieli/)")
    st.sidebar.markdown("---")
    
    with st.sidebar.expander("About this App"):
     st.write("""
        This simple app is showing how to "do a radiography to a CNN".
        Showed here there are different methods to visualize what is happening inside the convolutional neural network
     """)
    with st.sidebar.expander("Additional information"):
     st.write("""
     """)
    with st.sidebar.expander("Aknowledgment"):
     Aknowledgment()
    
    st.sidebar.markdown("---")
    st.sidebar.header("Settings")
    mod_app = st.sidebar.selectbox('Select model for the app:',
            ('AlexaNET', 'VGG16', 'VGG19'),
            help = 'select the model to use in the app. Default AlexNet'
            )
    alexa_idx =[0, 3, 6, 8, 10]
    VGG16_filt = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    VGG19_filt = [0, 2, 5, 7, 10, 12, 14, 16,19, 21, 23, 25, 28, 30, 32, 34]
    if mod_app == 'AlexaNET':
          pret_mod= model
          filt_idx =conv_layer_indices(pret_mod)
    if mod_app == 'VGG16':
          pret_mod= VGG16()
          filt_idx =conv_layer_indices(pret_mod)
    if mod_app == 'VGG19':
          pret_mod= VGG19()
          filt_idx =conv_layer_indices(pret_mod)
    st.sidebar.markdown("---")
    st.sidebar.markdown("Some of the methods required you select a convolutional layer and a filter")
    conv_layer_app = st.sidebar.selectbox(
        'Select one convolution layer', filt_idx,
        help = 'select convolutional filter layer')    
    conv_layer_app = int(conv_layer_app)
    x = pret_mod.eval()
    max = x.features[conv_layer_app].out_channels -1
    filter_app = st.sidebar.slider('select one filter', 0, max, 1)
    st.sidebar.markdown("---")
    st.sidebar.markdown("Some of the methods required you select a layer")
    max = len(pret_mod.eval().features) -1
    Layer_app = st.sidebar.slider('select one target layer', 0, max, 1,
                        help= 'select target layer of the model')    
    

    #### Main Corpus ####

    methods_tab, theory_tab = st.tabs(["CNNscan", "Theory & details"])
    with methods_tab:
        st.write("")

    with st.expander("Visualize the structure"):
        if  mod_app == 'AlexaNET':
            url1 = "https://github.com/SalvatoreRa/CNNscan/blob/main/img/alexnet.png?raw=true"
            url2 = "https://github.com/SalvatoreRa/CNNscan/blob/main/img/alexnet2.png?raw=true"
            response = requests.get(url1)
            img_screen = Image.open(BytesIO(response.content))
            st.image(img_screen)
            response = requests.get(url2)
            img_screen = Image.open(BytesIO(response.content))
            st.image(img_screen)
        st.write('visualize the structure as a dataframe')
        show_df = st.button('show the dataframe', help= 'visualize the structure in a dataframe')
        if show_df:
            model_layers_to_df(pret_mod)
        show_structure =st.button('show the structure as a graph',
         help= 'visualize the structure as a graph')
        if show_structure:
            plot_conv_model_structure(pret_mod)
            


    with st.expander("Visualize the filters"):
        st.write('Default model is **AlexNet** which is faster')
        
        st.markdown("---")
        st.markdown("Please select on the sidebar the convolutional layer")
        st.markdown("The first 16 filters of the selected convolutional layer will be visualized")
        show_filters = st.button('show the filters')
        if show_filters:
            fetch_filters(pret_mod, filt_idx, layer = conv_layer_app)
            
    with st.expander("Alternative visualization of the filters"):
        st.write('Default model is **AlexNet** which is faster')
        
        st.markdown("---")
        st.markdown("Please select on the sidebar the convolutional layer and a specific filter")
        st.markdown("This method will return six images for the selected filter (the images are obtained at different steps)")
        show_alt_filters = st.button('visualize the filter')
        if show_alt_filters:
            imgs_filt =advance_filt(pret_mod, conv_layer_app, filter_app )
            output_adv_filt(imgs_filt)
            buf = BytesIO()
            imgs_filt[5].save(buf, format="JPEG")
            byte_im =buf.getvalue()
            st.download_button(
                label="Download Last Image",
                data=byte_im,
                file_name="alternative_filters"+".jpg",
                mime="image/jpg",
                key = 'alternative_filters_download_button'
                )
            

    with st.expander("Visualize the feature maps"):
        st.write('Default model is **AlexNet** which is faster')
        
        st.markdown("---")
        st.markdown("Please select an image for the method")
        st.markdown("This method will return the feature maps")
        image_to_use = st.selectbox(
        'Select the image to use',
        ('provided test', 'provide image'),
        help = 'select the image to test. You can use the provided image or upload an image (jpg, png)')

        if image_to_use == 'provide image':
            image_features = load_test_image()
        else:
            image_features = load_baseline()

        show_featuremaps = st.button('show the feature maps')
        if show_featuremaps:
            fetch_feature_maps(pret_mod, image_features)

    with st.expander("Visualize GradCam"):
        st.write('Default model is **AlexNet** which is faster')
        
        st.markdown("---")
        st.markdown("Please select a target layer on the sidebar")
        st.markdown("This method will return the heatmap, superimposed images")
        
        image_to_cam = st.selectbox(
        'Select an image to use',
        ('provided test', 'provide image'),
        help = 'select the image to test. You can use the provided image or upload an image (jpg, png)')

        if image_to_cam == 'provide image':
            image_cam = load_test_image()
        else:
            image_cam = load_baseline()

        show_gradcam = st.button('show GradCam')
        if show_gradcam:
            
            heats, sup, act_map = Visualize_GradCam(pret_mod, image_cam, target_layer=Layer_app)
            cam_outputs(image_cam, heats, sup, act_map)
            
    with st.expander("Visualize Vanilla Backpropagation"):
        st.write('Default model is **AlexNet** which is faster, however other models leads to different results')
        
        

        image_to_backpr = st.selectbox(
        'Select an image for Vanilla Backpropagation:',
        ('provided test', 'provide image'),
        help = 'select the image to test. You can use the provided image or upload an image (jpg, png)')

        if image_to_backpr == 'provide image':
            image_to_backpr = load_test_image()
        else:
            image_to_backpr = load_baseline()

        show_backprop = st.button('show Vanilla Backpropagation')
        if show_backprop:
            
            backprop_im, backprop_bn =VanillaBackprop_process(pret_mod, image_to_backpr)
            
            txt1 = 'Original image' 
            txt2 = 'Colored Vanilla Backpropagation'
            txt3 = 'Vanilla Backpropagation Saliency'
            outputs_backprop(image_to_backpr, backprop_im, backprop_bn, 
                             txt1, txt2, txt3)
            
    with st.expander("Visualize guided Backpropagation"):
      
        image_to_Gbackpr = st.selectbox(
        'Select an image for Guided Backpropagation:',
        ('provided test', 'provide image'),
        help = 'select the image to test. You can use the provided image or upload an image (jpg, png)')

        if image_to_Gbackpr == 'provide image':
            image_to_Gbackpr = load_test_image()
        else:
            image_to_Gbackpr = load_baseline()

        show_Gbackprop = st.button('show Guided Backpropagation')
        if show_Gbackprop:
            
            Gbackprop_im, Gbackprop_bn, pos_sal_bp, neg_sal_bp =GuidedBackprop_process(pret_mod, image_to_Gbackpr)
            
            txt1 = 'Original image' 
            txt2 = 'Guided Backpropagation Negative Saliency'
            txt3 = 'Guided Backpropagation Saliency'
            outputs_backprop(image_to_Gbackpr, Gbackprop_im, Gbackprop_bn, 
                             txt1, txt2, txt3)
            
            txt1 = 'Original image' 
            txt2 = 'Guided Backpropagation Negative Saliency'
            txt3 = 'Guided Backpropagation Positive Saliency'
            outputs_backprop(image_to_Gbackpr, pos_sal_bp, neg_sal_bp, 
                             txt1, txt2, txt3)
            
    with st.expander("Visualize ScoreCam"):
      
        image_to_scorecam = st.selectbox(
        'Select an image for ScoreCam:',
        ('provided test', 'provide image'),
        help = 'select the image to test. You can use the provided image or upload an image (jpg, png)')

        if image_to_scorecam == 'provide image':
            image_to_scorecam = load_test_image()
        else:
            image_to_scorecam = load_baseline()

        show_scorecam = st.button('show ScoreCam')
        if show_scorecam:
          scorecam = scorecam_process(pret_mod, image_to_scorecam)
          heatmap, heatmap_on_image, activation_map = save_class_activation_images(image_to_scorecam, scorecam)
          txt1 = 'Original image' 
          txt2 = 'Score-weighted Class Activation Map colored'
          txt3 = 'Score-weighted Class Activation Map on image'
          txt4 = 'Score-weighted Class Activation Map - black and white'
          outputs_scorecam(image_to_scorecam, heatmap, heatmap_on_image, activation_map, txt1, txt2, txt3, txt4)
          
    with st.expander("Visualize Integrated Gradients"):
      
        image_to_grad = st.selectbox(
        'Select an image for Integrated Gradients:',
        ('provided test', 'provide image'),
        help = 'select the image to test. You can use the provided image or upload an image (jpg, png)')

        if image_to_grad == 'provide image':
            image_to_grad = load_test_image()
        else:
            image_to_grad = load_baseline()

        show_int_grad = st.button('show Integrated Gradient')
        if show_int_grad:
            
            im, im_bn = integrated_gradient_process(image_to_grad, pret_mod)
            
            txt1 = 'Original image' 
            txt2 = 'Colored Integrated Gradient'
            txt3 = 'Integrated Gradient in grays'
            outputs_backprop(image_to_grad, im, im_bn, 
                             txt1, txt2, txt3)
            
    with st.expander("Visualize guided GradCam"):
      
        image_to_GGradCam = st.selectbox(
        'Select an image for Guided GradCam:',
        ('provided test', 'provide image'),
        help = 'select the image to test. You can use the provided image or upload an image (jpg, png)')

        if image_to_GGradCam == 'provide image':
            image_to_GGradCam = load_test_image()
        else:
            image_to_GGradCam = load_baseline()

        show_GGradCam = st.button('show Guided GradCam')
        if show_GGradCam:
            
            cam_im, cam_gs = gradient_gradcam(pret_mod, image_to_GGradCam)
            
            txt1 = 'Original image' 
            txt2 = 'Guided GradCam Colors'
            txt3 = 'Guided GradCam grayscale'
            outputs_backprop(image_to_GGradCam, cam_im, cam_gs, 
                             txt1, txt2, txt3)   

    with st.expander("Visualize Layerwise Relevance"):
      
        image_to_LRP = st.selectbox(
        'Select an image for Layerwise Relevance:',
        ('provided test', 'provide image'),
        help = 'select the image to test. You can use the provided image or upload an image (jpg, png)')

        if image_to_LRP == 'provide image':
            image_to_LRP = load_test_image()
        else:
            image_to_LRP = load_baseline()

        show_LRP = st.button('show Layerwise Relevance')
        if show_LRP:
            heat_list =LRP_process(pret_mod, image_to_LRP)
            outputs_LRP(image_to_LRP, heat_list)
            
    with st.expander("Visualize LayerCAM"):
        
        image_to_layerCAM = st.selectbox(
        'Select an image for LayerCAM:',
        ('provided test', 'provide image'),
        help = 'select the image to test. You can use the provided image or upload an image (jpg, png)')
        st.markdown("---")
        st.markdown("Please select on the target layer")
        st.markdown("This method will return the heatmap, superimposed images")
        
        if image_to_layerCAM == 'provide image':
            image_to_layerCAM = load_test_image()
        else:
            image_to_layerCAM = load_baseline()
                
    
        
        show_LayerCAM = st.button('show LayerCAM')
        if show_LayerCAM:
            heatmap, heatmap_on_image, activation_map = LayerCAM_process(image_to_layerCAM, 
                                                                    pret_mod, layer =Layer_app)
            txt1 = 'Original image' 
            txt2 = 'Class Activation Map - layerCAM, layer: ' + str(Layer_app)
            txt3 = 'Class Activation HeatMap - layerCAM, layer: ' + str(Layer_app)
            txt4 = 'layerCAM usperimposed on the image, layer: ' + str(Layer_app)
            outputs_scorecam(image_to_layerCAM, activation_map, heatmap, heatmap_on_image, txt1, txt2, txt3, txt4)
            
    with st.expander("Grad Times Image"):
      
        image_to_GTI = st.selectbox(
        'Select an image for Grad Times Imagee:',
        ('provided test', 'provide image'),
        help = 'select the image to test. You can use the provided image or upload an image (jpg, png)')

        if image_to_GTI == 'provide image':
            image_to_GTI = load_test_image()
        else:
            image_to_GTI = load_baseline()

        show_GTI = st.button('show Grad Times Images')
        if show_GTI:
            grad_times_image, grayscale_vanilla_grads, BackProg_times_image, grayscale_BackProg_grads, integrated_grads_times, grayscale_int_grads_times = Grad_times_process(image_to_GTI, pret_mod)
            txt1 = 'Original image' 
            txt2 = 'Colored Vanilla x Gradient image'
            txt3 = 'gray-scale Vanilla x Gradient image'
            outputs_backprop(image_to_GTI, grad_times_image, grayscale_vanilla_grads, 
                             txt1, txt2, txt3)
            txt1 = 'Original image' 
            txt2 = 'Colored backpropagated x Gradient image'
            txt3 = 'gray-scale backpropagated x Gradient image'
            outputs_backprop(image_to_GTI, BackProg_times_image, grayscale_BackProg_grads, 
                             txt1, txt2, txt3)
            txt1 = 'Original image' 
            txt2 = 'Colored integrated x Gradient image'
            txt3 = 'gray-scale integrated x Gradient image'
            outputs_backprop(image_to_GTI, integrated_grads_times, grayscale_int_grads_times, 
                             txt1, txt2, txt3)
          
    with st.expander("Smooth Grad Image"):
      
        image_to_SGI = st.selectbox(
        'Select an image for Smooth Grad Imagee:',
        ('provided test', 'provide image'),
        help = 'select the image to test. You can use the provided image or upload an image (jpg, png)')

        if image_to_SGI == 'provide image':
            image_to_SGI = load_test_image()
        else:
            image_to_SGI = load_baseline()

        show_VSGI = st.button('show Vanilla Smooth Grad Images')
        if show_VSGI:
            smooths, smooths_bn = smooth_grad_process(image_to_SGI, pret_mod)
            outputs_smoothgrad(image_to_SGI, smooths, smooths_bn, desc= 'Vanilla Backprop.')
        
        show_GSGI = st.button('show Guided Smooth Grad Images')
        
        if show_GSGI:
            smooths, smooths_bn = smooth_grad_process_guidBackprop(image_to_SGI, pret_mod)
            outputs_smoothgrad(image_to_SGI, smooths, smooths_bn, desc= 'Guided Backprop.')
     
    
    with st.expander("Layer activation with guided backpropagation"):
        st.write('Default model is **AlexNet** which is faster')
        st.write('If you want to know more check: [Filter visualization](https://github.com/SalvatoreRa/CNNscan/blob/main/addendum.md#filter-visualization)')
        st.markdown("---")
        st.markdown("Please select on the sidebar the convolutional layer and a specific filter")


        image_to_LAGB = st.selectbox(
        'Select an image for layer activation:',
        ('provided test', 'provide image'),
        help = 'select the image to test. You can use the provided image or upload an image (jpg, png)')

        if image_to_LAGB == 'provide image':
            image_to_LAGB = load_test_image()
        else:
            image_to_LAGB = load_baseline()
            
            
        show_layer_act_guid_bp = st.button('visualize the Layer activation')
        if show_layer_act_guid_bp:
            imgs_layr =layer_act_guid_bp(image_to_LAGB, pret_mod, conv_layer_app, filter_app)
            output_layer_act_guid_bp(imgs_layr, image_to_LAGB)
            
    with st.expander("Inverted Image Representations"):
        st.write('Default model is **AlexNet** which is faster')
        st.write('If you want to know more check: [Filter visualization](https://github.com/SalvatoreRa/CNNscan/blob/main/addendum.md#filter-visualization)')
        st.markdown("---")
        st.markdown("Please select on the sidebar a layer")


        image_to_invert = st.selectbox(
        'Select an image for Inverted Image Representations:',
        ('provided test', 'provide image'),
        help = 'select the image to test. You can use the provided image or upload an image (jpg, png)',
        key = 'image_to_invert')

        if image_to_invert == 'provide image':
            image_to_invert = load_test_image()
        else:
            image_to_invert = load_baseline()
            
        image_size = st.radio(
            "select the image size:",
            ('224', '256', '512'),
            key ='inverted image size',
            help = 'dimension of the inverted image (squared, ex = 224,224)')
        image_size= int(image_size)
            
        show_image_invert= st.button('visualize the Inverted Image Representations')
        if show_image_invert:
            
            imgs_invert =inverted_representation_process(image_to_invert, pret_mod, image_size, Layer_app )
            output_inverted(imgs_invert, image_to_invert)
            buf = BytesIO()
            imgs_invert[8].save(buf, format="JPEG")
            byte_im =buf.getvalue()
            st.download_button(
                label="Download Last Image",
                data=byte_im,
                file_name="styled_img"+".jpg",
                mime="image/jpg",
                key = 'inverted image  download'
                )



    with st.expander("Class generated images"):
        st.write('Default model is **AlexNet** which is faster')
        st.write('If you want to know more check: [Filter visualization](https://github.com/SalvatoreRa/CNNscan/blob/main/addendum.md#filter-visualization)')
        st.markdown("---")
        st.markdown("Please select a class image")

        col1, col2, col3 = st.columns( [0.2, 0.6, 0.2])
        with col1:
            print("place holder otherwise he gave an error")
        with col2:
            categ_imagenet = st.cache(read_imagenet_categ)()
            st.write('### Imagene categories', categ_imagenet)
        with col3:
            print("place holder otherwise he gave an error")

        st.markdown("---")
        
        class_to_gen = st.slider('select one class', 0, 999, 1,
                        help= 'select the corrisponding number of the image class you want')
        
        class_to_gen = int(class_to_gen)
        st.markdown("---")

        col1, col2 = st.columns( [0.2, 0.8])
        with col1:
            st.write(class_to_gen)
        with col2:
            st.write(categ_imagenet.iloc[class_to_gen, 0])


                   
            
        class_gen_img= st.button('visualize class generated images')
        if class_gen_img:
            
            imgs_gen = class_generated_images(pret_mod, class_to_gen)
            outputs_CGI(imgs_gen)
            buf = BytesIO()
            imgs_gen[8].save(buf, format="JPEG")
            byte_im =buf.getvalue()
            st.download_button(
                label="Download Last Image",
                data=byte_im,
                file_name="styled_img"+".jpg",
                mime="image/jpg",
                key = 'class generated  download'
                )
        
        
        reg_gen_choice = st.selectbox(
        'Select parameters for regularized model:',
        ('default', 'customize'),
        help = 'you can use the default parameters or set the parameter you prefer',
        key = 'reg_gen_choice button')

        if reg_gen_choice == 'default':
            target_class = 130  # Flamingo
            iterations=150
            blur_freq=4
            blur_rad=1
            wd=0.0001
            clipping_value=0.1
            initial_learning_rate = 6
        else:
            target_class = class_to_gen
            iterations = st.slider('select number iterations', 50, 300, 1) 
            blur_freq = st.slider('select blur frequency', 1, 10, 1)
            blur_rad = st.slider('select blur radius', 1, 10, 1)
            wd= st.selectbox('Select layer:', 
                ('0.00001', '0.00005', '0.0001', '0.0005', '0.001', 
                '0.005', '0.01', '0.05','0.1'))
            wd = float(wd)
            clipping_value= st.slider('select clipping value', 0.1, 0.9, 0.1)
            initial_learning_rate = st.slider('select initial learning rate', 1, 10, 1)
            
            
        class_gen_reg_img= st.button('visualize class generated images',
                                     help = 'This is the regularized version',
                                     key = 'regularized version')
        if class_gen_reg_img:
        
            imgs_gen_reg = regularized_class_img_gen(pret_mod, target_class, iterations, blur_freq, blur_rad, wd, clipping_value, initial_learning_rate)
            outputs_CGI(imgs_gen_reg)
            buf = BytesIO()
            imgs_gen_reg[8].save(buf, format="JPEG")
            byte_im =buf.getvalue()
            st.download_button(
                label="Download Last Image",
                data=byte_im,
                file_name="styled_img"+".jpg",
                mime="image/jpg",
                key = 'class generated regularized download'
                )    

    with st.expander("DeepDream"):

        st.write('Default model is **AlexNet** which is faster, however other models leads to better results')
        st.markdown("---")
        st.markdown("Please select on the sidebar the convolutional layer and a specific filter")
        image_to_DD = st.selectbox(
        'Select an image for DeepDream:',
        ('provided test', 'provide image'),
        help = 'select the image to test. You can use the provided image or upload an image (jpg, png)')

        if image_to_DD == 'provide image':
            image_to_DD = load_test_image()
        else:
            image_to_DD = load_baseline()

 
        show_DD = st.button('show DeepDream')
        if show_DD:
            images_dd = dream(mod_dd, conv_layer_app, filter_app, image_to_DD)       
            outputs_DD(images_dd)
            buf = BytesIO()
            images_dd[11].save(buf, format="JPEG")
            byte_im =buf.getvalue()
            st.download_button(
                label="Download Last Image",
                data=byte_im,
                file_name="styled_img"+".jpg",
                mime="image/jpg"
                )
    with theory_tab:
        with st.expander("Overview of a CNN"):
            CNN_overview()

if __name__ == "__main__":
    main()
