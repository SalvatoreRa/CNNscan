#structural libraries
import streamlit as st
import io
from PIL import Image
from io import BytesIO
import requests

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
import os
import sys
import pathlib

##########################################################
###########         Visualize Gradcam      ###############
##########################################################


def cam_outputs(image_cam, heats, sup, act_map):
    col1, col2 = st.columns([0.33, 0.33])
    with col1:
        st.write('Original image')
        st.image(image_cam)
        st.write('activation map')
        st.image(act_map)
    with col2:
        st.write('Correspective heatmap')
        st.image(heats)
        st.write('Superimposed image')
        st.image(sup)

##########################################################
########### Visualize vanilla propagation  ###############
##########################################################

def outputs_backprop(im1, im2, im3, txt1, txt2, txt3):
    col1, col2, col3 = st.columns([0.25, 0.25, 0.25])
    with col1:
        st.write(txt1)
        st.image(im1)
    with col2:
        st.write(txt2)
        st.image(im2)
    with col3:
        st.write(txt3)
        st.image(im3)

##########################################################
###########         Visualize SCORE-CAM    ###############
##########################################################


def outputs_scorecam(im1, im2, im3, im4, txt1, txt2, txt3, txt4):
    col1, col2 = st.columns([0.25, 0.25])
    with col1:
        st.write(txt1)
        st.image(im1)
        st.write(txt3)
        st.image(im3)
    with col2:
        st.write(txt2)
        st.image(im2)
        st.write(txt4)
        st.image(im4)

##########################################################
###########  Visualize Guided SCORE-CAM    ###############
##########################################################

##########################################################
###########Visualize Layerwise Relevance LRP##############
##########################################################

def outputs_LRP(img, heat_list):
    col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])
    with col1:
        st.write('Original image')
        st.image(img)
        st.write('Layer 4')
        st.image(heat_list[3])
        st.write('Layer 8')
        st.image(heat_list[7])
    with col2:
        st.write('Layer 1')
        st.image(heat_list[0])
        st.write('Layer 5')
        st.image(heat_list[4])
        st.write('Layer 9')
        st.image(heat_list[8])
    with col3:
        st.write('Layer 2')
        st.image(heat_list[1])
        st.write('Layer 6')
        st.image(heat_list[5])
        st.write('Layer 10')
        st.image(heat_list[9])
    with col4:
        st.write('Layer 3')
        st.image(heat_list[2])
        st.write('Layer 7')
        st.image(heat_list[6])
        st.write('Layer 11')
        st.image(heat_list[10])

##########################################################
###########     Visualize Smooth Grad      ###############
##########################################################


def outputs_smoothgrad(img, smooths, smooths_bn, desc= 'Vanilla Backprop.'):
    col1, col2, col3,  = st.columns([0.33, 0.33, 0.33])
    with col1:
        st.write('Original image')
        st.image(img)
        st.write('Colored ' + desc + ' sigma: 3')
        st.image(smooths[2])
        st.write('Original image')
        st.image(img)
        st.write('Grayscale ' + desc + ' sigma: 3')
        st.image(smooths_bn[2])
    with col2:
        st.write('Colored ' + desc + ' sigma: 1')
        st.image(smooths[0])
        st.write('Colored ' + desc + ' sigma: 4')
        st.image(smooths[3])
        st.write('Grayscale ' + desc + ' sigma: 1')
        st.image(smooths_bn[0])
        st.write('Grayscale ' + desc + ' sigma: 4')
        st.image(smooths_bn[3])
    with col3:
        st.write('Colored ' + desc + ' sigma: 2')
        st.image(smooths[1])
        st.write('Colored ' + desc + ' sigma: 5')
        st.image(smooths[4])
        st.write('Grayscale ' + desc + ' sigma: 2')
        st.image(smooths_bn[1])
        st.write('Grayscale ' + desc + ' sigma: 5')
        st.image(smooths_bn[4])



##########################################################
###########  Visualize advanced Filters    ###############
##########################################################

def output_adv_filt(images):
    col1, col2, col3= st.columns([0.33, 0.33, 0.33])
    with col1:
        st.image(images[0])
        st.image(images[3])
        
    with col2:
        st.image(images[1])
        st.image(images[4])

    with col3:
        st.image(images[2])
        st.image(images[5])

##########################################################
###########  Layer activation              ###############
###########  with guided backpropagation   ###############
##########################################################

def output_layer_act_guid_bp(imgs_layr, img):
    col1, col2, col3= st.columns([0.33, 0.33, 0.33])
    with col1:
        st.write('original image')
        st.image(img)
        st.write('positive saliency')
        st.image(imgs_layr[2])
        
    with col2:
        st.write('colored gradient')
        st.image(imgs_layr[0])
        st.write('negative saliency')
        st.image(imgs_layr[3])

    with col3:
        st.write('black and white gradient')
        st.image(imgs_layr[1])

##########################################################
###########    Inverted representation     ###############
##########################################################

def output_inverted(images, img):
    col1, col2, col3= st.columns([0.33, 0.33, 0.33])
    st.write('the first image is the original image, the others method generated')
    with col1:
        st.image(img)
        st.image(images[3])
        st.image(images[6])
        
    with col2:
        st.image(images[1])
        st.image(images[4])
        st.image(images[7])

    with col3:
        st.image(images[2])
        st.image(images[5])
        st.image(images[8])

##########################################################
###########    Class generated images      ###############
##########################################################

def outputs_CGI(images):
    col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])
    with col1:
        st.image(images[0])
        st.image(images[4])
        st.image(images[8])
    with col2:
        st.image(images[1])
        st.image(images[5])
        st.image(images[9])
    with col3:
        st.image(images[2])
        st.image(images[6])
        
    with col4:
        st.image(images[3])
        st.image(images[7])
        
        
##########################################################
###########         Visualize DeepDream    ###############
##########################################################

def outputs_DD(images):
    col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])
    with col1:
        st.image(images[0])
        st.image(images[4])
        st.image(images[8])
    with col2:
        st.image(images[1])
        st.image(images[5])
        st.image(images[9])
    with col3:
        st.image(images[2])
        st.image(images[6])
        st.image(images[10])
    with col4:
        st.image(images[3])
        st.image(images[7])
        st.image(images[11])


##########################################################
###########         Visualize LIME         ###############
##########################################################

def outputs_LIME(images):
    col1, col2, col3 = st.columns([0.33, 0.33, 0.33])
    with col1:
        st.write("original image")
        st.image(images[0])
    with col2:
        st.write("positive association with image")
        st.image(images[1])
    with col3:
        st.write("pos/neg association with image")
        st.image(images[2])