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
###########     Visualize Smooth Grad      ###############
##########################################################


def generate_smooth_grad(Backprop, prep_img, target_class, param_n, param_sigma_multiplier):

    smooth_grad = np.zeros(prep_img.size()[1:])

    mean = 0
    sigma = param_sigma_multiplier / (torch.max(prep_img) - torch.min(prep_img)).item()
    for x in range(param_n):

        noise = Variable(prep_img.data.new(prep_img.size()).normal_(mean, sigma**2))
        noisy_img = prep_img + noise
        vanilla_grads = Backprop.generate_gradients(noisy_img, target_class)
        smooth_grad = smooth_grad + vanilla_grads
    smooth_grad = smooth_grad / param_n
    return smooth_grad

@st.cache(ttl=12*3600)
def smooth_grad_process(img, model):
  im, pred_cls = process_img(img, model)
  param_n = 50
  VBP = VanillaBackprop(model)
  smooths = list()
  smooths_bn = list()
  for param_sigma in range(1,6):
    
    smooth_grad = generate_smooth_grad(VBP, im, pred_cls, param_n, param_sigma)
    smooth_grad_bn = convert_to_grayscale(smooth_grad)
    smooth_grad = save_gradient_images(smooth_grad)
    smooth_grad_bn = save_gradient_images(smooth_grad_bn)
    smooths.append(smooth_grad)
    smooths_bn.append(smooth_grad_bn)
  return smooths, smooths_bn

@st.cache(ttl=12*3600)
def smooth_grad_process_guidBackprop(img, model):
  im, pred_cls = process_img(img, model)
  param_n = 50
  GBP = GuidedBackprop(model)
  smooths = list()
  smooths_bn = list()
  for param_sigma in range(1,6):
    
    smooth_grad = generate_smooth_grad(GBP, im, pred_cls, param_n, param_sigma)
    smooth_grad_bn = convert_to_grayscale(smooth_grad)
    smooth_grad = save_gradient_images(smooth_grad)
    smooth_grad_bn = save_gradient_images(smooth_grad_bn)
    smooths.append(smooth_grad)
    smooths_bn.append(smooth_grad_bn)
  return smooths, smooths_bn

##########################################################
###########  Visualize advanced Filters    ###############
##########################################################

