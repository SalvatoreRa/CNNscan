<img src="https://github.com/SalvatoreRa/CNNscan/blob/main/img/cnn_scan.png?raw=true" width="600" height="600"/>

# CNNscan
A CT-scan of your CNN

<img src="https://github.com/SalvatoreRa/CNNscan/blob/main/img/logo.png?raw=true" width="400" height="200"/>


# Introduction

So far, in almost every course on computer vision I have seen that only three methods are presented to be able to visualize and inspect a convolutional neural network (CNN): filter visualization, feature maps and gradCAM.

In reality, many other methods have been created and studied over the years. Visualizing what a neural network sees is critical to understanding how it functions and how it interprets the world. That is why I decided to develop a web app that can allow you to test different methods at the click of a button. 

The app allows you to change parameters, select different layers, choose filters and other options (including select import images). In addition, it is currently possible to view three different models. This allows you to understand how the CNN is extracting features from your input data.

The app also also features over 20 methods, this app will help you gain a deeper understanding of CNN models and how they work. 

In the future, several more methods will be added, increased the number of models featured (possibly uploading a user-trained model) and the theoretical description of the methods. 

**Over to you to play!**

link to the app:

<a href="https://salvatorera-cnnscan-cnnscan-y7c8pc.streamlit.app/">
  <img src="https://github.com/SalvatoreRa/CNNscan/blob/main/img/logo.png?raw=true" width="200" height="100"/>
</a>


# Implemented methods
* filter visualization - implemented with AlexaNet, VGG16, VGG19
* feature map visualization - implemented with AlexaNet, VGG16, VGG19
* GradCam - implemented with AlexaNet, VGG16, VGG19
* Colored Vanilla Backpropagation
* Vanilla Backpropagation Saliency
* Colored Guided Backpropagation and Saliency
* Guided Backpropagation, negative and positive saliency
* Score-Cam
* Guided GradCam
* Layerwise Relevance
* LayerCAM
* Grad Times Images
* Smooth Grad
* Deep Dream (with alexnet, VGG16, VGG19)
* enhanced filter visualization
* Layer activation
* Inverted image representation
* Class Specific Image Generation
* LIME, SHAP value
* implementation other models for different methods (VGG16, VGG19...)

# On their way 
* implementation other models for different methods (LeNet...)
* better download option
* theoretical description
* and more

Stay tuned!




![work in progress](https://github.com/SalvatoreRa/CNNscan/blob/main/img/work_in_progress.png?raw=true)
