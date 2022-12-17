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

def fetch_filters(layer = 0):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    model.eval()
    print('done downloading')
    idx_conv_layer = [0, 3, 6, 8, 10]
    filters = []
    for layer_idx in idx_conv_layer:
        filters.append(model.features[layer_idx].weight.data)
    t = filters[layer]
    fig = plt.figure(figsize=(4,4))
    num_rows = 4
    num_cols = 4
    if layer == 0:
      for i in range(16):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        npimg = np.array(t[i].numpy(), np.float32)

        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis('off')
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([]) 
      plt.tight_layout()
      plt.show()
    else:
      limit = 16
      plt_dim = int(math.sqrt(limit))
      for i, filt in enumerate(filters[layer].numpy()[:limit]):
          plt.subplot(plt_dim, plt_dim, i+1)
          plt.imshow(filt[0], cmap="gray")
          plt.axis('off')
      plt.show()

# Create the main app
def main():
    show_filters = st.button('show the filters')
    if show_filters:
        fetch_filters(layer = 0)
    


if __name__ == "__main__":
    main()
