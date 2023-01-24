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


#this to import modules
sys.path.append(str(pathlib.Path().absolute()).split("/src")[0] + "/src")
from utils import load_test_image, load_baseline,  \
    format_np_output, save_image, save_gradient_images, convert_to_grayscale, \
    process_img, save_class_activation_images, scorecam_process, \
    apply_colormap_on_image, apply_heatmap, recreate_image, \
    preprocess_image
from methods import fetch_filters, fetch_feature_maps, CamExtractor, \
    GradCam, Visualize_GradCam, VanillaBackprop, VanillaBackprop_process
from outputs import cam_outputs, outputs_backprop

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
















##########################################################
###########  Visualize vanilla propagation ###############
##########################################################

#this code is adapted from: https://github.com/utkuozbulak/pytorch-cnn-visualizations

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        def relu_backward_hook_function(module, grad_in, grad_out):
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            self.forward_relu_outputs.append(ten_out)
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):
        model_output = self.model(input_image)
        self.model.zero_grad()
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        model_output.backward(gradient=one_hot_output)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr

def get_positive_negative_saliency(gradient):
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency
 
def GuidedBackprop_process(model, img):
  GuideProg = GuidedBackprop(model)
  im, pred_cls = process_img(img, model)
  gradient = GuideProg.generate_gradients(im, pred_cls)
  grad_im =save_gradient_images(gradient)
  grad_bn= convert_to_grayscale(gradient)
  grad_im_bn =save_gradient_images(grad_bn)
  pos_sal, neg_sal = get_positive_negative_saliency(gradient)
  pos_sal =save_gradient_images(pos_sal)
  neg_sal =save_gradient_images(neg_sal)
  return grad_im, grad_im_bn, pos_sal, neg_sal

##########################################################
###########         Visualize SCORE-CAM    ###############
##########################################################

#this code is adapted from: https://github.com/utkuozbulak/pytorch-cnn-visualizations

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.model.classifier(x)
        return conv_output, x


class ScoreCam():
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        target = conv_output[0]
        cam = np.ones(target.shape[1:], dtype=np.float32)
        for i in range(len(target)):
            saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :],0),0)
            saliency_map = F.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
            if saliency_map.max() == saliency_map.min():
                continue
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            w = F.softmax(self.extractor.forward_pass(input_image*norm_saliency_map)[1],dim=1)[0][target_class]
            cam += w.data.numpy() * target[i, :, :].data.numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        return cam





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

#this code is adapted from: https://github.com/utkuozbulak/pytorch-cnn-visualizations

def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb

class CamExtractor():

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.model.classifier(x)
        return conv_output, x


class GradCam():
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        target = conv_output.data.numpy()[0]
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        return cam

@st.cache(ttl=12*3600)      
def gradient_gradcam(model, img):
  gcv2 = GradCam(model, target_layer=11)
  im, pred_cls = process_img(img, model)
  cam = gcv2.generate_cam(im, pred_cls)
  GBP = GuidedBackprop(model)
  guided_grads = GBP.generate_gradients(im, pred_cls)
  cam_gb = guided_grad_cam(cam, guided_grads)
  cam_im =save_gradient_images(cam_gb)
  cam_gs = convert_to_grayscale(cam_gb)
  cam_gs =save_gradient_images(cam_gs)
  return cam_im, cam_gs


##########################################################
###########Visualize Layerwise Relevance LRP##############
##########################################################
# Visualize Layerwise Relevance LRP
#this code is adapted from: https://github.com/utkuozbulak/pytorch-cnn-visualizations


class LRP():
    def __init__(self, model):
        self.model = model

    def LRP_forward(self, layer, input_tensor, gamma=None, epsilon=None):

        if gamma is None:
            gamma = lambda value: value + 0.05 * copy.deepcopy(value.data.detach()).clamp(min=0)
        if epsilon is None:
            eps = 1e-9
            epsilon = lambda value: value + eps
        layer = copy.deepcopy(layer)

        try:
            layer.weight = nn.Parameter(gamma(layer.weight))
        except AttributeError:
            pass

        try:
            layer.bias = nn.Parameter(gamma(layer.bias))
        except AttributeError:
            pass


        return epsilon(layer(input_tensor))

    def LRP_step(self, forward_output, layer, LRP_next_layer):

        forward_output = forward_output.requires_grad_(True)

        lrp_rule_forward_out = self.LRP_forward(layer, forward_output, None, None)

        ele_div = (LRP_next_layer / lrp_rule_forward_out).data

        (lrp_rule_forward_out * ele_div).sum().backward()

        LRP_this_layer = (forward_output * forward_output.grad).data

        return LRP_this_layer

    def generate(self, input_image, target_class):
        layers_in_model = list(self.model._modules['features']) + list(self.model._modules['classifier'])
        number_of_layers = len(layers_in_model)

        features_to_classifier_loc = len(self.model._modules['features'])

        forward_output = [input_image]

        for conv_layer in list(self.model._modules['features']):
            forward_output.append(conv_layer.forward(forward_output[-1].detach()))

        feature_to_class_shape = forward_output[-1].shape

        forward_output[-1] = torch.flatten(forward_output[-1], 1)
        for index, classifier_layer in enumerate(list(self.model._modules['classifier'])):
            forward_output.append(classifier_layer.forward(forward_output[-1].detach()))

        target_class_one_hot = torch.FloatTensor(1, 1000).zero_()
        target_class_one_hot[0][target_class] = 1

        LRP_per_layer = [None] * number_of_layers + [(forward_output[-1] * target_class_one_hot).data]

        for layer_index in range(1, number_of_layers)[::-1]:

            if layer_index == features_to_classifier_loc-1:
                LRP_per_layer[layer_index+1] = LRP_per_layer[layer_index+1].reshape(feature_to_class_shape)

            if isinstance(layers_in_model[layer_index], (torch.nn.Linear, torch.nn.Conv2d, torch.nn.MaxPool2d)):

                lrp_this_layer = self.LRP_step(forward_output[layer_index], layers_in_model[layer_index], LRP_per_layer[layer_index+1])
                LRP_per_layer[layer_index] = lrp_this_layer
            else:
                LRP_per_layer[layer_index] = LRP_per_layer[layer_index+1]
        return LRP_per_layer




def LRP_process(model, img):
  layerwise_relevance = LRP(model)
  im, pred_cls = process_img(img, model)
  LRP_per_layer = layerwise_relevance.generate(im, pred_cls)
  heat_list = list()
  for layer in range(1,12):
      lrp_to_vis = np.array(LRP_per_layer[layer][0]).sum(axis=0)
      lrp_to_vis = np.array(Image.fromarray(lrp_to_vis).resize((im.shape[2],
                              im.shape[3]), Image.ANTIALIAS))
      heatmap = apply_heatmap(lrp_to_vis, 4, 4)
      heat_list.append(heatmap)
  return heat_list



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
###########         Visualize LayerCAM     ###############
##########################################################

#this code is adapted from: https://github.com/utkuozbulak/pytorch-cnn-visualizations

class CamExtractor():

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):

        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.model.classifier(x)
        return conv_output, x


class LayerCam():

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        target = conv_output.data.numpy()[0]
        weights = guided_gradients
        weights[weights < 0] = 0 # discard negative gradients
        cam = np.sum(weights * target, axis=0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255

        return cam


def LayerCAM_process(img, model, layer =1):
  im, pred_cls = process_img(img, model)
  layer_cam = LayerCam(model, target_layer=layer)
  cam = layer_cam.generate_cam(im, pred_cls)
  heatmap, heatmap_on_image, activation_map = save_class_activation_images(img, cam)
  return heatmap, heatmap_on_image, activation_map


##########################################################
########### Visualize Integrated Gradients ###############
##########################################################

#this code is adapted from: https://github.com/utkuozbulak/pytorch-cnn-visualizations

class IntegratedGradients():
    """
        Produces gradients generated with integrated gradients from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.model.eval()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def generate_images_on_linear_path(self, input_image, steps):
        step_list = np.arange(steps+1)/steps
        xbar_list = [input_image*step for step in step_list]
        return xbar_list

    def generate_gradients(self, input_image, target_class):
        model_output = self.model(input_image)
        self.model.zero_grad()
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        model_output.backward(gradient=one_hot_output)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr

    def generate_integrated_gradients(self, input_image, target_class, steps):
        xbar_list = self.generate_images_on_linear_path(input_image, steps)
        integrated_grads = np.zeros(input_image.size())
        for xbar_image in xbar_list:
            single_integrated_grad = self.generate_gradients(xbar_image, target_class)
            integrated_grads = integrated_grads + single_integrated_grad/steps
        return integrated_grads[0]

@st.cache(ttl=12*3600)
def integrated_gradient_process(img, model):
    im, pred_cls = process_img(img, model)
    IG = IntegratedGradients(model)
    integrated_grads = IG.generate_integrated_gradients(im, pred_cls, 100)
    grayscale_integrated_grads = convert_to_grayscale(integrated_grads)
    im = save_gradient_images(integrated_grads)
    im_bn = save_gradient_images(grayscale_integrated_grads)
    return im, im_bn


##########################################################
###########  Visualize Grad Times images   ###############
##########################################################

#this code is adapted from: https://github.com/utkuozbulak/pytorch-cnn-visualizations

@st.cache(ttl=12*3600)    
def Grad_times_process(img, model):
  im, pred_cls = process_img(img, model)
  VBP = VanillaBackprop(model)
  vanilla_grads = VBP.generate_gradients(im, pred_cls)
  grad_times_image = vanilla_grads * im.detach().numpy()[0]
  grayscale_vanilla_grads = convert_to_grayscale(grad_times_image)
  grad_times_image = save_gradient_images(grad_times_image)
  grayscale_vanilla_grads = save_gradient_images(grayscale_vanilla_grads)

  GuideProg = GuidedBackprop(model)
  BackProg_grads = GuideProg.generate_gradients(im, pred_cls)
  BackProg_times_image = BackProg_grads * im.detach().numpy()[0]
  grayscale_BackProg_grads = convert_to_grayscale(BackProg_times_image)
  BackProg_times_image = save_gradient_images(BackProg_times_image)
  grayscale_BackProg_grads = save_gradient_images(grayscale_BackProg_grads)

  IG = IntegratedGradients(model)
  integrated_gradient = IG.generate_integrated_gradients(im, pred_cls, 100)
  integrated_grads_times = integrated_gradient * im.detach().numpy()[0]
  grayscale_int_grads_times = convert_to_grayscale(integrated_grads_times)
  integrated_grads_times = save_gradient_images(integrated_grads_times)
  grayscale_int_grads_times = save_gradient_images(grayscale_int_grads_times)
  return grad_times_image, grayscale_vanilla_grads, BackProg_times_image, grayscale_BackProg_grads, integrated_grads_times, grayscale_int_grads_times


##########################################################
###########     Visualize Smooth Grad      ###############
##########################################################

#this code is adapted from: https://github.com/utkuozbulak/pytorch-cnn-visualizations

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
class CNNLayerVisualization():
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
      

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            self.conv_output = grad_out[0, self.selected_filter]
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        self.hook_layer()
        random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        processed_image = preprocess_image(random_image, False)
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        images = list()
        for i in range(1, 31):
            optimizer.zero_grad()
            x = processed_image
            for index, layer in enumerate(self.model):
                x = layer(x)
                if index == self.selected_layer:
                    break
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            loss.backward()
            optimizer.step()
            self.created_image = recreate_image(processed_image)
            if i % 5 == 0:
                im = save_image(self.created_image)
                images.append(im)
        return images
        
    def visualise_layer_without_hooks(self):
        random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        processed_image = preprocess_image(random_image, False)
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        images = list()
        for i in range(1, 31):
            optimizer.zero_grad()
            x = processed_image
            for index, layer in enumerate(self.model):
                x = layer(x)
                if index == self.selected_layer:
                    break
            self.conv_output = x[0, self.selected_filter]
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            loss.backward()
            optimizer.step()
            self.created_image = recreate_image(processed_image)
            if i % 5 == 0:
                im =save_image(self.created_image)
                images.append(im)
        return images



def advance_filt(mod, cnn_layer, filter_pos ):
  layer_vis = CNNLayerVisualization(mod.features, cnn_layer, filter_pos)
  images = layer_vis.visualise_layer_with_hooks()
  return images

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

class GuidedBackprop():
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        def relu_backward_hook_function(module, grad_in, grad_out):
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            self.forward_relu_outputs.append(ten_out)

        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class, cnn_layer, filter_pos):
        self.model.zero_grad()
        x = input_image
        for index, layer in enumerate(self.model.features):
            x = layer(x)
            if index == cnn_layer:
                break
        conv_output = torch.sum(torch.abs(x[0, filter_pos]))
        conv_output.backward()
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr

def layer_act_guid_bp(img, model, cnn_layer, filter_pos):
  im, pred_cls = process_img(img, model)
  GBP = GuidedBackprop(model)
  guided_grads = GBP.generate_gradients(im, pred_cls, cnn_layer, filter_pos)
  col_grad_img =save_gradient_images(guided_grads)
  grayscale_guided_grads = convert_to_grayscale(guided_grads)
  grayscale_guided_grads =save_gradient_images(grayscale_guided_grads)
  pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
  pos_sal =save_gradient_images(pos_sal)
  neg_sal =save_gradient_images(neg_sal)
  return [col_grad_img, grayscale_guided_grads, pos_sal, neg_sal]

def output_layer_act_guid_bp(images, img):
    col1, col2, col3= st.columns([0.33, 0.33, 0.33])
    with col1:
        st.write('original image')
        st.image(img)
        st.write('positive saliency')
        st.image(images[2])
        
    with col2:
        st.write('colored gradient')
        st.image(images[0])
        st.write('negative saliency')
        st.image(images[3])

    with col3:
        st.write('black and white gradient')
        st.image(images[1])
        



##########################################################
###########         Visualize DeepDream    ###############
##########################################################


class DeepDream():
    def __init__(self, model, selected_layer, selected_filter, image):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.created_image = image.convert('RGB')        
        self.hook_layer()
    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            self.conv_output = grad_out[0, self.selected_filter]


        self.model[self.selected_layer].register_forward_hook(hook_function)

    def dream(self):

        self.processed_image = preprocess_image(self.created_image, True)
        optimizer = SGD([self.processed_image], lr=12,  weight_decay=1e-4)
        images = list()
        for i in range(1, 251):
            optimizer.zero_grad()
            x = self.processed_image
            for index, layer in enumerate(self.model):
                x = layer(x)
                if index == self.selected_layer:
                    break
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            loss.backward()
            optimizer.step()
            self.created_image = recreate_image(self.processed_image)
            if i % 20 == 0:
                print(self.created_image.shape)
                im =save_image(self.created_image)
                images.append(im)
        return images

@st.cache(ttl=3600)
def VGG16():
    pret_mod =  models.vgg16(pretrained=True)
    return pret_mod

@st.cache(ttl=3600)
def VGG19():
    pret_mod =  models.vgg19(pretrained=True)
    return pret_mod

def dream(model, cnn_layer, filter_pos, image):
    dd = DeepDream(model.features, cnn_layer, filter_pos, image)
    images = dd.dream()
    return images
    
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

    st.sidebar.image(logo,  width=150)
    st.sidebar.markdown("Made by [Salvatore Raieli](https://www.linkedin.com/in/salvatore-raieli/)")
    st.sidebar.markdown("---")
    
    with st.sidebar.expander("About this App"):
     st.write("""
        This simple app is showing how to "do a radiography to a CNN".
        //
        Showed here there are different methods to visualize what is happening inside the convolutional neural network
     """)
    with st.sidebar.expander("Additional information"):
     st.write("""
     """)
    with st.sidebar.expander("Aknowledgment"):
     st.write("""
     """)
    
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
          filt_idx =alexa_idx
          pret_mod= model
    if mod_app == 'VGG16':
          filt_idx =VGG16_filt
          pret_mod= VGG16()
    if mod_app == 'VGG19':
          filt_idx =VGG19_filt
          pret_mod= VGG19()
    
        
        
    

    with st.expander("Visualize the structure"):
        url1 = "https://github.com/SalvatoreRa/CNNscan/blob/main/img/alexnet.png?raw=true"
        url2 = "https://github.com/SalvatoreRa/CNNscan/blob/main/img/alexnet2.png?raw=true"
        response = requests.get(url1)
        img_screen = Image.open(BytesIO(response.content))
        st.image(img_screen)
        response = requests.get(url2)
        img_screen = Image.open(BytesIO(response.content))
        st.image(img_screen)

    with st.expander("Visualize the filters"):
        st.write('Default model is **AlexNet** which is faster')
        st.write('If you want to know more check: [Filter visualization](https://github.com/SalvatoreRa/CNNscan/blob/main/addendum.md#filter-visualization)')
        
        conv_layer = st.selectbox(
        'Select the convolution layer', filt_idx,
        help = 'select convolutional filter layer')
        option = int(conv_layer)
        show_filters = st.button('show the filters')
        if show_filters:
            fetch_filters(pret_mod, filt_idx, layer = option)
            
    with st.expander("Alternative visualization of the filters"):
        st.write('Default model is **AlexNet** which is faster')
        st.write('If you want to know more check: [Filter visualization](https://github.com/SalvatoreRa/CNNscan/blob/main/addendum.md#filter-visualization)')
        
        conv_layer_alt = st.selectbox(
        'Select a convolution layer', filt_idx,
        help = 'select convolutional filter layer')
        option = int(conv_layer_alt)
        x = pret_mod.eval()
        max = x.features[option].out_channels -1
        filter_pos_alt = st.slider('select filter', 0, max, 1)
        show_alt_filters = st.button('visualize the filter')
        if show_alt_filters:
            imgs_filt =advance_filt(pret_mod, option, filter_pos_alt )
            output_adv_filt(imgs_filt)

    with st.expander("Visualize the feature maps"):
        st.write('Default model is **AlexNet** which is faster')
        st.write('If you want to know more check: [Filter visualization](https://github.com/SalvatoreRa/CNNscan/blob/main/addendum.md#feature-map-visualization)')
        
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
        st.write('If you want to know more check: [Filter visualization](https://github.com/SalvatoreRa/CNNscan/blob/main/addendum.md#GradCam)')
        
        max = len(pret_mod.eval().features) -1
        target_layer = st.slider('select target layer', 0, max, 1,
                        help= 'select target layer of the model')
        t = target_layer

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
            
            heats, sup, act_map = Visualize_GradCam(pret_mod, image_cam, target_layer=t)
            cam_outputs(image_cam, heats, sup, act_map)
            
    with st.expander("Visualize Vanilla Backpropagation"):
        st.write('Default model is **AlexNet** which is faster, however other models leads to different results')
        st.write('If you want to know more check: [Filter visualization](https://github.com/SalvatoreRa/CNNscan/blob/main/addendum.md#Vanilla-Backpropagation)')
        

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
        
        
        if image_to_layerCAM == 'provide image':
            image_to_layerCAM = load_test_image()
        else:
            image_to_layerCAM = load_baseline()
                
        max = len(pret_mod.eval().features) -1
        Layer = st.slider('select taret layer', 0, max, 1,
                        help= 'select target layer of the model')
        
        show_LayerCAM = st.button('show LayerCAM')
        if show_LayerCAM:
            heatmap, heatmap_on_image, activation_map = LayerCAM_process(image_to_layerCAM, 
                                                                    pret_mod, layer =Layer)
            txt1 = 'Original image' 
            txt2 = 'Class Activation Map - layerCAM, layer: ' + str(Layer)
            txt3 = 'Class Activation HeatMap - layerCAM, layer: ' + str(Layer)
            txt4 = 'layerCAM usperimposed on the image, layer: ' + str(Layer)
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
        
        conv_layer_gb = st.selectbox(
        'Select  a convolution layer', filt_idx,
        help = 'select convolutional filter layer')
        option = int(conv_layer_gb)
        x = pret_mod.eval()
        max = x.features[option].out_channels -1
        filter_pos_gb = st.slider('select filter', 0, max, 1)
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
            imgs_layr =layer_act_guid_bp(image_to_LAGB, pret_mod, option, filter_pos_gb)
            output_layer_act_guid_bp(imgs_layr, image_to_LAGB)

    with st.expander("DeepDream"):

        st.write('Default model is **AlexNet** which is faster, however other models leads to better results')
      
        image_to_DD = st.selectbox(
        'Select an image for DeepDream:',
        ('provided test', 'provide image'),
        help = 'select the image to test. You can use the provided image or upload an image (jpg, png)')

        if image_to_DD == 'provide image':
            image_to_DD = load_test_image()
        else:
            image_to_DD = load_baseline()

        DD_par = st.selectbox(
        'Select parameters for DD:',
        ('default', 'customize'),
        help = 'you can use the default parameters (model, layer, filter) or set the parameter you prefer')

        if DD_par == 'default':
            mod_dd = model
            cnn_layer = 10
            filter_pos = 8

        else:
            mod_dd = st.selectbox('Select model for DeepDream:',
            ('AlexaNET', 'VGG16', 'VGG19'),
             help = 'select the model you want to use. VGG16 and VGG19 are computationally expensive and they will take time')
            if mod_dd == 'AlexaNET':
                mod_dd = model
                cnn_layer = st.selectbox('Select layer:', 
                                         ('0', '3', '6', '8', '10'),
                                        help = 'select the convolutional layer')
                cnn_layer = int(cnn_layer)
            if mod_dd == 'VGG16':
                pret_mod = VGG16()
                mod_dd = pret_mod
                cnn_layer = st.selectbox('Select layer:', 
                ('0', '2', '5', '7', '10', '12', '14', '17',
                '19', '21', '24', '26', '28' ),
                 help = 'select the convolutional layer')
                cnn_layer = int(cnn_layer)
            if mod_dd == 'VGG19':
                pret_mod = VGG19()
                mod_dd = pret_mod
                cnn_layer = st.selectbox('Select layer:', 
                ('0', '2', '5', '7', '10', '12', '14', '16',
                '19', '21', '23', '25', '28', '30',
                '32', '34'),
                help = 'select the convolutional layer')
                cnn_layer = int(cnn_layer)
            
            x = mod_dd.eval()
            max = x.features[cnn_layer].out_channels -1
            filter_pos = st.slider('select filter', 0, max, 1)
            


                
            

        show_DD = st.button('show DeepDream')
        if show_DD:
            images_dd = dream(mod_dd, cnn_layer, filter_pos, image_to_DD)       
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

if __name__ == "__main__":
    main()
