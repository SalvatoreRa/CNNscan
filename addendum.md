
<img src="https://github.com/SalvatoreRa/CNNscan/blob/main/img/cnn_scan.png?raw=true" width="600" height="600"/>


# CNNscan
A CT-scan of your CNN

# CNN overview

**[Convolutional nets](https://en.wikipedia.org/wiki/Convolutional_neural_network)** showed to outperform classical neural network on image datasets. A convolutional layer takes an image and as output has a 3D tensor of shape (height, width, channels). Interesting, the height and width are less than original image (shrinkage that increase going deep in the network). The convolutional neural network following the convolutional layers there is at least one dense layer. Generally, the last convolutional layer is flattened before to going through the dense layers. The principal difference between dense layer and convolutional layer is that the first learn global patterns while the second learns local pattern of an image.

![cnn](https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Typical_cnn.png/395px-Typical_cnn.png)

Convolutional layers have two important characteristics:

* **They are translational invariant**, if they learn a certain pattern (ex. in a corner) they can recognize anywhere (ex. at the center of the image). Instead a dense net would have to learn again the pattern if it appears in a new location. Therefore, convnets are data efficient in processing images (since translation invariance is important in images, convnets require less example to learn representation).
* **Spatial hierarchies of pattern**, the first layer in a convnets is learning pattern as edges, while the second layer will learn pattern using the combinations of features learnt by the first layer. This is important because going deep in the network convnets learn increasingly complex pattern and more abstract.

![CNN architecture](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Comparison_image_neural_networks.svg/512px-Comparison_image_neural_networks.svg.png)

# filter visualization
<img src="https://github.com/SalvatoreRa/CNNscan/blob/main/img/alexa_filters.png?raw=true" width="858" height="375"/>
# feature map visualization

# GradCam

# Vanilla Backpropagation

# Vanilla Backpropagation Saliency

# Colored Guided Backpropagation and Saliency

# Guided Backpropagation, negative and positive saliency

# Score-Cam

# Guided GradCam

# Layerwise Relevance

# LayerCAM

# Grad Times Images

# Smooth Grad




![work in progress](https://github.com/SalvatoreRa/CNNscan/blob/main/img/work_in_progress.png?raw=true)
