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


sys.path.append(str(pathlib.Path().absolute()).split("/src")[0] + "/src")
from utils import load_test_image, load_baseline,  \
    format_np_output, save_image, save_gradient_images, convert_to_grayscale, \
    process_img, save_class_activation_images, scorecam_process, \
    apply_colormap_on_image, apply_heatmap, recreate_image, \
    preprocess_image, get_positive_negative_saliency, \
    guided_grad_cam

#part of this code is adapted from: https://github.com/utkuozbulak/pytorch-cnn-visualizations        
# check his amazing repository

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
###########  Visualize vanilla propagation ###############
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


