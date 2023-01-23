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

