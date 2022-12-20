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
from sklearn.preprocessing import minmax_scale
from matplotlib import cm
from torch.nn import ReLU
from torch.autograd import Variable
import torch.nn.functional as F
import copy
from matplotlib.colors import ListedColormap
from copy import deepcopy

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


#Fetch filters
def fetch_filters(model, layer = 0):
    
    idx_conv_layer = [0, 3, 6, 8, 10]
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

### Feature Maps
def load_test_image():
    uploaded_file = st.file_uploader(label='Upload an image for test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def load_baseline():
    img_path = "https://github.com/SalvatoreRa/CNNscan/blob/main/img/manja-vitolic-gKXKBY-C-Dk-unsplash.jpg?raw=true"
    response = requests.get(img_path)
    img_screen = Image.open(BytesIO(response.content))
    st.image(img_screen)
    return img_screen

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

# Visualize GradCam
def visualize_gradcam(model, img):
  shapes = (np.array(img).shape[1], np.array(img).shape[0])

  norm_mean = [0.485, 0.456, 0.406]
  norm_std = [0.229, 0.224, 0.225]

  data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
  im = data_transform(img)

  output = model(im.unsqueeze(0))
  _, pred_cls = output.max(dim=1, keepdim=True)
  def generate_heatmap(output, class_id, model, image):
    
    prediction = model(image)
    prediction = torch.max(prediction)
    prediction.backward()
    gradients = model.gradients
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.get_activations(image)
    for i in range(256):
      activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.detach().numpy(), 0)
    heatmap = torch.from_numpy(heatmap)
    heatmap /= torch.max(heatmap)
    return heatmap
  heatmap = generate_heatmap(output, pred_cls, model, im.unsqueeze(0))
  heat = Image.fromarray(np.uint8(cm.jet(heatmap.numpy())*255))
  heats = heat.resize(shapes, Image.ANTIALIAS)
  sup =  Image.blend(img.convert("RGBA"), heats, 0.5)
  return heats, sup

def outputs(image_cam, heats, sup):
    col1, col2, col3 = st.columns([0.25, 0.25, 0.25])
    with col1:
        st.write('Original image')
        st.image(image_cam)
    with col2:
        st.write('Correspective heatmap')
        st.image(heats)
    with col3:
        st.write('Superimposed image')
        st.image(sup)


        
        
        
# Visualize vanilla propagation
#this code is adapted from: https://github.com/utkuozbulak/pytorch-cnn-visualizations

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
  
  output = model( im.unsqueeze(0))
  _, pred_cls = output.max(dim=1, keepdim=True)
  im_proc = Variable( im.unsqueeze(0), requires_grad=True)
  return im_proc, pred_cls


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

# Visualize vanilla propagation
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

# Visualize SCORE-CAM
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

def save_class_activation_images(org_img, activation_map):
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    activation_map = save_image(activation_map )
    return heatmap, heatmap_on_image, activation_map

def scorecam_process(model, img):
  im, pred_cls = process_img(img, model)
  score_cam = ScoreCam(model, target_layer=11)
  cam = score_cam.generate_cam(im, pred_cls)
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

# Visualize Guided SCORE-CAM
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

def LRP_process(model, img):
  layerwise_relevance = LRP(model)
  im, pred_cls = process_img(img, model)
  LRP_per_layer = layerwise_relevance.generate(im, pred_cls)
  heat_list = []
  for layer in range(1,12):
      lrp_to_vis = np.array(LRP_per_layer[layer][0]).sum(axis=0)
      lrp_to_vis = np.array(Image.fromarray(lrp_to_vis).resize((im.shape[2],
                              im.shape[3]), Image.ANTIALIAS))
      heatmap = apply_heatmap(lrp_to_vis, 4, 4)
      heat_list.append(heatmap)
  return heatmap


def outputs_LRP(img, heat_list):
col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])
    with col1:
        st.write('Original image')
        st.image(im1)
        st.write('Layer 4')
        st.image(heat_list[3])
    with col2:
        st.write('Layer 1')
        st.image(heat_list[0])
        st.write('Layer 5')
        st.image(heat_list[4])
          

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
    with st.sidebar.expander("About this App"):
     st.write("""
        This simple app is showing how to "do a radiography to a CNN".
        //
        Showed here there are different methods to visualize what is happening inside the convolutional neural network
     """)
    with st.sidebar.expander("About AlexNet"):
        st.write("""
        
        """)
    with st.sidebar.expander("About visualize filters"):
        st.write("""
        
        """)
    with st.sidebar.expander("About visualize feature maps"):
        st.write("""
        
        """)
    with st.sidebar.expander("About GradCam"):
        st.write("""
        
        """)
    with st.sidebar.expander("Vanilla Backpropagation"):
        st.write("""
        
        """)
    with st.sidebar.expander("Guided Backpropagation"):
        st.write("""
        
        """)
    with st.sidebar.expander("ScoreCam"):
        st.write("""
        
        """)
    with st.sidebar.expander("Guided GradCam"):
        st.write("""
        
        """)
    with st.sidebar.expander("Layerwise Relevance"):
        st.write("""
        
        """)    

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
        conv_layer = st.selectbox(
        'Select the convolution layer',
        ('0', '3', '6', '8','10'))
        option = int(conv_layer)
        show_filters = st.button('show the filters')
        if show_filters:
            fetch_filters(model, layer = option)

    with st.expander("Visualize the feature maps"):
        image_to_use = st.selectbox(
        'Select the image to use',
        ('provided test', 'provide image'))

        if image_to_use == 'provide image':
            image_features = load_test_image()
        else:
            image_features = load_baseline()

        show_featuremaps = st.button('show the feature maps')
        if show_featuremaps:
            fetch_feature_maps(model, image_features)

    with st.expander("Visualize GradCam"):
        image_to_cam = st.selectbox(
        'Select an image to use',
        ('provided test', 'provide image'))

        if image_to_cam == 'provide image':
            image_cam = load_test_image()
        else:
            image_cam = load_baseline()

        show_gradcam = st.button('show GradCam')
        if show_gradcam:
            heats, sup =visualize_gradcam(model, image_cam)
            outputs_prog(image_cam, heats, sup)
            
    with st.expander("Visualize Vanilla Backpropagation"):
      
        image_to_backpr = st.selectbox(
        'Select an image for Vanilla Backpropagation:',
        ('provided test', 'provide image'))

        if image_to_backpr == 'provide image':
            image_to_backpr = load_test_image()
        else:
            image_to_backpr = load_baseline()

        show_backprop = st.button('show Vanilla Backpropagation')
        if show_backprop:
            
            backprop_im, backprop_bn =VanillaBackprop_process(model, image_to_backpr)
            
            txt1 = 'Original image' 
            txt2 = 'Colored Vanilla Backpropagation'
            txt3 = 'Vanilla Backpropagation Saliency'
            outputs_backprop(image_to_backpr, backprop_im, backprop_bn, 
                             txt1, txt2, txt3)
            
    with st.expander("Visualize guided Backpropagation"):
      
        image_to_Gbackpr = st.selectbox(
        'Select an image for Guided Backpropagation:',
        ('provided test', 'provide image'))

        if image_to_Gbackpr == 'provide image':
            image_to_Gbackpr = load_test_image()
        else:
            image_to_Gbackpr = load_baseline()

        show_Gbackprop = st.button('show Guided Backpropagation')
        if show_Gbackprop:
            
            Gbackprop_im, Gbackprop_bn, pos_sal_bp, neg_sal_bp =GuidedBackprop_process(model, image_to_Gbackpr)
            
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
        ('provided test', 'provide image'))

        if image_to_scorecam == 'provide image':
            image_to_scorecam = load_test_image()
        else:
            image_to_scorecam = load_baseline()

        show_scorecam = st.button('show ScoreCam')
        if show_scorecam:
          scorecam = scorecam_process(model, image_to_scorecam)
          heatmap, heatmap_on_image, activation_map = save_class_activation_images(image_to_scorecam, scorecam)
          txt1 = 'Original image' 
          txt2 = 'Score-weighted Class Activation Map colored'
          txt3 = 'Score-weighted Class Activation Map on image'
          txt4 = 'Score-weighted Class Activation Map - black and white'
          outputs_scorecam(image_to_scorecam, heatmap, heatmap_on_image, activation_map, txt1, txt2, txt3, txt4)
            
    with st.expander("Visualize guided GradCam"):
      
        image_to_GGradCam = st.selectbox(
        'Select an image for Guided GradCam:',
        ('provided test', 'provide image'))

        if image_to_GGradCam == 'provide image':
            image_to_GGradCam = load_test_image()
        else:
            image_to_GGradCam = load_baseline()

        show_GGradCam = st.button('show Guided GradCam')
        if show_GGradCam:
            
            cam_im, cam_gs = gradient_gradcam(model, image_to_GGradCam)
            
            txt1 = 'Original image' 
            txt2 = 'Guided GradCam Colors'
            txt3 = 'Guided GradCam grayscale'
            outputs_backprop(image_to_GGradCam, cam_im, cam_gs, 
                             txt1, txt2, txt3)   

    with st.expander("Visualize Layerwise Relevance"):
      
        image_to_LRP = st.selectbox(
        'Select an image for Layerwise Relevance:',
        ('provided test', 'provide image'))

        if image_to_LRP == 'provide image':
            image_to_LRP = load_test_image()
        else:
            image_to_LRP = load_baseline()

        show_LRP = st.button('show Guided GradCam')
        if show_LRP:
            heat_list =LRP_process(model, image_to_LRP)
            outputs_LRP(image_to_LRP, heat_list)         


if __name__ == "__main__":
    main()
