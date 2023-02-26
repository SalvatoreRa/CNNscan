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
import networkx


sys.path.append(str(pathlib.Path().absolute()).split("/src")[0] + "/src")
from utils import (load_test_image, load_baseline, 
    format_np_output, save_image, save_gradient_images, convert_to_grayscale, 
    process_img, save_class_activation_images,  
    apply_colormap_on_image, apply_heatmap, recreate_image, 
    preprocess_image, get_positive_negative_saliency, 
    guided_grad_cam, preprocess_and_blur_image)

#part of this code is adapted from: https://github.com/utkuozbulak/pytorch-cnn-visualizations        
# check his amazing repository

def model_layers_to_df(model):
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            rows.append((name, str(module)))
    df = pd.DataFrame(rows, columns=["Name", "Layer"])
    return  st.dataframe(df)

##########################################################
###########         Fetch Filters          ###############
##########################################################


def fetch_filters(model, idx_conv_layer = [0, 3, 6, 8, 10], layer = 0):
    
    
    filters = []
    for layer_idx in idx_conv_layer:
        filters.append(model.features[layer_idx].weight.data)
    t = filters[idx_conv_layer.index(layer)]
    fig = plt.figure(figsize=(4,4))
    num_rows = 4
    num_cols = 4
    if layer == 0:
      plt_dim = int(math.sqrt(16))
      fig, axis =plt.subplots(plt_dim, plt_dim)
      ax = axis.flatten()
      for i in range(16):
        npimg = np.array(t[i].numpy(), np.float32)

        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax[i].imshow(npimg)
        ax[i].axis('off')
        ax[i].set_title(str(i))
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([]) 
      plt.tight_layout()
      st.pyplot(fig)
    else:
      plt_dim = int(math.sqrt(16))
      fig, axis =plt.subplots(plt_dim, plt_dim)
      ax = axis.flatten()
      for i in range(len(ax)):
        ax[i].imshow(filters[idx_conv_layer.index(layer)].numpy()[i][0], cmap="gray")
        ax[i].set_title(str(i))
        ax[i].axis('off')
      plt.tight_layout()
      st.pyplot(fig)


##########################################################
###########         Feature maps           ###############
##########################################################


def fetch_feature_maps(model, img):
  norm_mean = [0.485, 0.456, 0.406]
  norm_std = [0.229, 0.224, 0.225]

  data_transform = transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(256),
          transforms.ToTensor(),
          transforms.Normalize(norm_mean, norm_std),
      ])
  im = data_transform(img)

  def fetcher(image, model):
    model_children = list(list(model.children())[0])
    results=[model_children[0](image)]
    for i in range(1,len(model_children)):
      results.append(model_children[i](results[-1]))
    features = [results[i] for i in [2,5,12]]
    return features

  feature_maps = fetcher(im.unsqueeze(0), model)
  for num_layer in range(len(feature_maps)):
    layer_viz = feature_maps[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    st.write(layer_viz.size())
    fig, axis =plt.subplots(2, 8, figsize=(20, 10))
    ax = axis.flatten()
    for i in range(len(ax)):
        ax[i].imshow(layer_viz[i], cmap="gray")
        ax[i].set_title(str(i))
        ax[i].axis('off')
    
    st.pyplot(fig)
    plt.close()

##########################################################
###########         Visualize Gradcam      ###############
##########################################################


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
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
        weights = np.mean(guided_gradients, axis=(1, 2)) 
        cam = np.ones(target.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        return cam


def Visualize_GradCam(model, img, target_layer=11):
  grad_cam = GradCam(model, target_layer)
  im, pred_cls = process_img(img, model)
  cam = grad_cam.generate_cam(im, pred_cls)
  heatmap, heatmap_on_image, activation_map = save_class_activation_images(img, cam)
  return heatmap, heatmap_on_image, activation_map


##########################################################
########### Visualize vanilla propagation  ###############
##########################################################

class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
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

    def generate_gradients(self, input_image, target_class):
        model_output = self.model(input_image)
        self.model.zero_grad()
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        model_output.backward(gradient=one_hot_output)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr

def VanillaBackprop_process(model, img):
  VBP = VanillaBackprop(model)
  im, pred_cls = process_img(img, model)
  gradient = VBP.generate_gradients(im, pred_cls)
  grad_im =save_gradient_images(gradient)
  grad_bn= convert_to_grayscale(gradient)
  grad_im_bn =save_gradient_images(grad_bn)
  return grad_im, grad_im_bn


##########################################################
###########  Visualize Guided propagation  ###############
##########################################################


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


class scoreCamExtractor():
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
        self.extractor = scoreCamExtractor(self.model, target_layer)

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
    
def scorecam_process(model, img):
  im, pred_cls = process_img(img, model)
  score_cam = ScoreCam(model, target_layer=11)
  cam = score_cam.generate_cam(im, pred_cls)
  return cam

##########################################################
###########  Visualize Guided SCORE-CAM    ###############
##########################################################



class CamExtractor2():

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


class GuidedGradCam():
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.extractor = CamExtractor2(self.model, target_layer)

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
  gcv2 = GuidedGradCam(model, target_layer=11)
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


##########################################################
###########         Visualize LayerCAM     ###############
##########################################################

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
###########  Layer activation              ###############
###########  with guided backpropagation   ###############
##########################################################

class LR_GuidedBackprop():
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
  GBP = LR_GuidedBackprop(model)
  guided_grads = GBP.generate_gradients(im, pred_cls, cnn_layer, filter_pos)
  col_grad_img =save_gradient_images(guided_grads)
  grayscale_guided_grads = convert_to_grayscale(guided_grads)
  grayscale_guided_grads =save_gradient_images(grayscale_guided_grads)
  pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
  pos_sal =save_gradient_images(pos_sal)
  neg_sal =save_gradient_images(neg_sal)
  images = [col_grad_img, grayscale_guided_grads, pos_sal, neg_sal]
  return images

##########################################################
###########     Inverted representation    ###############
##########################################################

class InvertedRepresentation():
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def alpha_norm(self, input_matrix, alpha):
        alpha_norm = ((input_matrix.view(-1))**alpha).sum()
        return alpha_norm

    def total_variation_norm(self, input_matrix, beta):
        to_check = input_matrix[:, :-1, :-1]  # Trimmed: right - bottom
        one_bottom = input_matrix[:, 1:, :-1]  # Trimmed: top - right
        one_right = input_matrix[:, :-1, 1:]  # Trimmed: top - right
        total_variation = (((to_check - one_bottom)**2 +
                            (to_check - one_right)**2)**(beta/2)).sum()
        return total_variation

    def euclidian_loss(self, org_matrix, target_matrix):
        distance_matrix = target_matrix - org_matrix
        euclidian_distance = self.alpha_norm(distance_matrix, 2)
        normalized_euclidian_distance = euclidian_distance / self.alpha_norm(org_matrix, 2)
        return normalized_euclidian_distance

    def get_output_from_specific_layer(self, x, layer_id):
        layer_output = None
        for index, layer in enumerate(self.model.features):
            x = layer(x)
            if str(index) == str(layer_id):
                layer_output = x[0]
                break
        return layer_output

    def generate_inverted_image_specific_layer(self, input_image, img_size, target_layer=3):
        opt_img = Variable(1e-1 * torch.randn(1, 3, img_size, img_size), requires_grad=True)
        optimizer = SGD([opt_img], lr=1e4, momentum=0.9)
        input_image_layer_output = \
            self.get_output_from_specific_layer(input_image, target_layer)
        alpha_reg_alpha = 6
        alpha_reg_lambda = 1e-7
        tv_reg_beta = 2
        tv_reg_lambda = 1e-8
        images = list()

        for i in range(201):
            optimizer.zero_grad()
            output = self.get_output_from_specific_layer(opt_img, target_layer)
            euc_loss = 1e-1 * self.euclidian_loss(input_image_layer_output.detach(), output)
            reg_alpha = alpha_reg_lambda * self.alpha_norm(opt_img, alpha_reg_alpha)
            reg_total_variation = tv_reg_lambda * self.total_variation_norm(opt_img,
                                                                            tv_reg_beta)
            loss = euc_loss + reg_alpha + reg_total_variation
            loss.backward()
            optimizer.step()
            # Generate image every 25 iterations
            if i % 25 == 0:
                print('Iteration:', str(i), 'Loss:', loss.data.numpy())
                recreated_im = recreate_image(opt_img)
                im =save_image(recreated_im)
                images.append(im)

            if i % 40 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 1/10
        return images


def inverted_representation_process(img, model, image_size,target_layer):
  '''
  inverted representation
  '''
  im, pred_cls = process_img(img, model)
  inverted_representation = InvertedRepresentation(model)
  images = inverted_representation.generate_inverted_image_specific_layer(im, image_size,target_layer)
  return images


##########################################################
###########    Class generated images      ###############
##########################################################

@st.cache(ttl=3600, suppress_st_warning=True)
class ClassSpecificImageGeneration():

      
    def __init__(self, model, target_class):
        self.mean = [-0.485, -0.456, -0.406]
        self.std = [1/0.229, 1/0.224, 1/0.225]
        self.model = model
        self.model.eval()
        self.target_class = target_class
        self.created_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))

    def generate(self, iterations=150):
        initial_learning_rate = 6
        images = list()

        

        for i in range(1, iterations):

            
            self.processed_image = preprocess_image(self.created_image, False)
            optimizer = SGD([self.processed_image], lr=initial_learning_rate)
            output = self.model(self.processed_image)
            class_loss = -output[0, self.target_class]

            if i % 15 == 0 or i == iterations-1:
                print('Iteration:', str(i), 'Loss',
                      "{0:.2f}".format(class_loss.data.numpy()))
            self.model.zero_grad()
            class_loss.backward()
            optimizer.step()
            self.created_image = recreate_image(self.processed_image)
            if i % 15 == 0 or i == iterations-1:
                
                im =save_image(self.created_image)
                images.append(im)

        return images

@st.cache(ttl=3600, suppress_st_warning=True)
def class_generated_images(model, class_to_gen):
  csig = ClassSpecificImageGeneration(model, class_to_gen)
  images =csig.generate()
  return images

##########################################################
###########         Regularized            ###############
###########  Class generated images        ###############
##########################################################


@st.cache(ttl=3600, suppress_st_warning=True)
class RegularizedClassSpecificImageGeneration():
  
    def __init__(self, model, target_class, iterations, blur_freq, blur_rad, wd, clipping_value, initial_learning_rate):
        self.mean = [-0.485, -0.456, -0.406]
        self.std = [1/0.229, 1/0.224, 1/0.225]
        self.model = model
        self.model.eval()
        self.target_class = target_class
        self.iterations = iterations
        self.blur_freq = blur_freq
        self.blur_rad = blur_rad
        self.wd = wd
        self.clipping_value = clipping_value
        self.initial_learning_rate = initial_learning_rate
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))


    def generate(self):
   
        initial_learning_rate = self.initial_learning_rate
        target_class = self.target_class
        iterations = self.iterations
        blur_freq = self.blur_freq
        blur_rad = self.blur_rad
        wd = self.wd
        clipping_value = self.clipping_value
        images = list()
        for i in range(1, iterations):
           
            if i % blur_freq == 0:
                self.processed_image = preprocess_and_blur_image(
                    self.created_image, False, blur_rad)
            else:
                self.processed_image = preprocess_and_blur_image(
                    self.created_image, False)
            
            optimizer = SGD([self.processed_image],
                            lr=initial_learning_rate, weight_decay=wd)
           
            output = self.model(self.processed_image)
            
            class_loss = -output[0, self.target_class]

            if i in np.linspace(0, iterations, 10, dtype=int):
                print('Iteration:', str(i), 'Loss',
                      "{0:.2f}".format(class_loss.data.cpu().numpy()))
            
            self.model.zero_grad()
            
            class_loss.backward()

            if clipping_value:
                torch.nn.utils.clip_grad_norm(
                    self.model.parameters(), clipping_value)
            
            optimizer.step()
            
            self.created_image = recreate_image(self.processed_image)

            if i % 15 == 0 or i == iterations-1:
                # Save image
                
                im =save_image(self.created_image)
                images.append(im)

        return images




target_class = 130  # Flamingo
iterations=150
blur_freq=4
blur_rad=1
wd=0.0001
clipping_value=0.1
initial_learning_rate = 6
pretrained_model = models.alexnet(pretrained=True)
def regularized_class_img_gen(pretrained_model, target_class, iterations, blur_freq, blur_rad, wd, clipping_value, initial_learning_rate):
  csig = RegularizedClassSpecificImageGeneration(pretrained_model, target_class, iterations, blur_freq, blur_rad, wd, clipping_value, initial_learning_rate)
  images = csig.generate()
  return images

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



def dream(model, cnn_layer, filter_pos, image):
    dd = DeepDream(model.features, cnn_layer, filter_pos, image)
    images = dd.dream()
    return images
