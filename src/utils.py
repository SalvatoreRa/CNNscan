#structural libraries
import streamlit as st
import io
from PIL import Image, ImageFilter
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




def load_test_image():
    '''
    load a test image
    '''
    uploaded_file = st.file_uploader(label='Upload an image for test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def load_baseline():
    '''
    load the baseline image: our beloved and judgemental cat
    '''
    img_path = "https://github.com/SalvatoreRa/CNNscan/blob/main/img/manja-vitolic-gKXKBY-C-Dk-unsplash-resized.jpg?raw=true"
    response = requests.get(img_path)
    img_screen = Image.open(BytesIO(response.content))
    st.image(img_screen)
    return img_screen


def conv_layer_indices(model):
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(int(name.split(".")[1]))
    return conv_layers

def process_img(img, model):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    data_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ])
    im = data_transform(img)
    im = im.unsqueeze(0)
    output = model(im)
    _, pred_cls = output.max(dim=1, keepdim=True)
    im = Variable(im, requires_grad=True)
    return im, pred_cls

def format_np_output(np_arr):
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

def save_image(im):
  im = format_np_output(im)
  im = Image.fromarray(im)
  return im

def save_gradient_images(gradient):
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    im =save_image(gradient)
    return im

def convert_to_grayscale(im_as_arr):
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

def save_class_activation_images(org_img, activation_map):
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    activation_map = save_image(activation_map )
    return heatmap, heatmap_on_image, activation_map


def apply_colormap_on_image(org_im, activation, colormap_name):
    color_map = cm.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)

    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    shapes = (np.array(org_im).shape[1], np.array(org_im).shape[0])
    heatmap_on_image = heatmap.resize(shapes, Image.ANTIALIAS)
    heatmap_on_image =  Image.blend(org_im.convert("RGBA"), heatmap_on_image, 0.5)
    return no_trans_heatmap, heatmap_on_image

def apply_heatmap(R, sx, sy):
    b = 10*((np.abs(R)**3.0).mean()**(1.0/3))
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx, sy))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')
    plt.imshow(R, cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
    with BytesIO() as buffer:
      plt.savefig(buffer, format = "png")
      buffer.seek(0)
      image = Image.open(buffer)
      ar = np.asarray(image)
    return image

def recreate_image(im_as_var):
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im

def preprocess_image(pil_im, resize_im=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    im_as_ten = torch.from_numpy(im_as_arr).float()
    im_as_ten.unsqueeze_(0)
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

def get_positive_negative_saliency(gradient):
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency

def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb

def read_imagenet_categ():
    imagenet_cat = "https://raw.githubusercontent.com/SalvatoreRa/CNNscan/main/imagenet1000_clsidx_to_labels.txt"
    response = requests.get(imagenet_cat)
    data_text = response.text
    data = eval(data_text)

    categ = pd.DataFrame.from_dict(data, orient='index',
                        columns=['cat'])
    return categ

def preprocess_and_blur_image(pil_im, resize_im=True, blur_rad=None):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print(
                "could not transform PIL_img to a PIL Image object. Please check input.")
    
    if resize_im:
        pil_im.thumbnail((224, 224))

    if blur_rad:
        pil_im = pil_im.filter(ImageFilter.GaussianBlur(blur_rad))

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  
 
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]

    im_as_ten = torch.from_numpy(im_as_arr).float()

    im_as_ten.unsqueeze_(0)

    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

def download_images(images, captions, cols = 2, rows = 2,
    file_name= 'images'):
    n_img = len(images)
    figure, axis = plt.subplots(rows, cols)
    ax = axis.flatten()
    for i in range(n_img):
      ax[i].imshow(images[i])
      ax[i].axis('off')
      if captions is not None:
        ax[i].set_title(str(captions[i]))
    plt.tight_layout()
    plt.show()

