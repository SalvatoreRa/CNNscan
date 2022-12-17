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


def fetch_filters():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    model.eval()
    idx_conv_layer = [0, 3, 6, 8, 10]
    filters = []
    for layer_idx in conv_layer_idx:
        filters.append(model.features[layer_idx].weight.data)

    limit = 16
    plt_dim = int(math.sqrt(limit))
    for i, filt in enumerate(filters[0].numpy()[:limit]):
        plt.subplot(plt_dim, plt_dim, i+1)
        plt.imshow(filt[0], cmap="gray")
        plt.axis('off')
    plt.show()

# Create the main app
def main():
    


if __name__ == "__main__":
    main()
