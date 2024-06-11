# 🎦 Experimental-VGG19-on-CIFAR10

## ✏️ Introduction

Performing CIFAR10 Image Classification using transfer learning and fine-tuning on preloaded VGG19 model in TensorFlow Keras and using customized VGG19 without weight initialization.

Using Machine Learning models to perform image classification into 10 categories. Based on the CIFAR-10 dataset from [University of Toronto CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

## 📦 Installation

The following packages are required to run the code:

- [Pandas](https://pandas.pydata.org/)
- [Numpy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [sklearn](https://scikit-learn.org/stable/)
- [TensorFlow](https://www.tensorflow.org/)

You should also have some sort of Python environment manager installed, such as [Anaconda](https://www.anaconda.com/).

## 🎯 Included Models:

1. Preloaded VGG19 with 'imagenet' weights (No Trainable Layers)
2. Preloaded VGG19 with 'imagenet' weights (Fully Trainable Model)
3. Custom VGG19 without weight initialization (Fully Trainable Model)

## 🏗️ Dataset:

Dataset is well-known CIFAR-10 dataset. It was retrieved in two packages: Training and Testing.
Testing package was 50-50 randomized to get validation and testing datasets.
Further data augmentation was performed on the training dataset to achieve final datasets.

1. Training Dataset: 50,000 images (excluding augmented data) --> 200,000 images (including augmented data)
2. Validation Dataset: 5,000 images
3. Testing Dataset: 5,000 images
