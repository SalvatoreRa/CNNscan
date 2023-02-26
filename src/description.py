#structural libraries
import streamlit as st
import io
from PIL import Image, ImageFilter
from io import BytesIO
import requests
import os
import sys
import pathlib

def Aknowledgment():
    st.markdown("""Many of these methods have been adapated from:
    [this repository](https://github.com/utkuozbulak/pytorch-cnn-visualizations).
    check it to better understand the code and how the method is working.
    """)


