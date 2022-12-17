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

def load_model():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    model.eval()
    return model

#Fetch filters
def fetch_filters(model, layer = 0):
    
    idx_conv_layer = [0, 3, 6, 8, 10]
    filters = []
    for layer_idx in idx_conv_layer:
        filters.append(model.features[layer_idx].weight.data)
    t = filters[layer]
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
        ax[i].imshow(filters[layer].numpy()[i][0], cmap="gray")
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


# Create the main app
def main():
    model = load_model()

    with st.sidebar.expander("About this App"):
     st.write("""
        This simple app is showing how to "do a radiography to a CNN".
        //
        Showed here there are different methods to visualize what is happening inside the convolutional neural network
     """)
    with st.sidebar.expander("About AlexNet"):
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

    with st.expander("Visualize the filters"):
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


if __name__ == "__main__":
    main()
