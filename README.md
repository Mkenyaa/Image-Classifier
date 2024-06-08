# Flower Species Classifier using Deep Learning

This repository contains code to develop an AI application that classifies different species of flowers using deep learning techniques. The application is designed to train an image classifier to recognize various types of flowers and can be used as part of a larger software system, such as a smartphone app, to identify flowers through images.

## Dataset
The project utilizes the [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html), which consists of 102 different categories of flowers. The dataset is divided into three parts: training, validation, and testing.

## Project Overview
The project is divided into the following main steps:

1. **Data Loading and Preprocessing**: The dataset is loaded using `torchvision` and preprocessed. Transformations such as random scaling, cropping, and flipping are applied to the training data to help the network generalize better. The images are resized to 224x224 pixels as required by the pre-trained networks. Mean and standard deviation normalization is also performed.

2. **Building and Training the Classifier**: A pre-trained network, VGG16, is used to extract image features. A new feed-forward classifier is defined and trained using these features. The training process includes backpropagation and optimization to minimize the loss and improve accuracy on the validation set.

3. **Testing the Network**: The trained network is evaluated on the test dataset to measure its performance on unseen data. The accuracy achieved on the test set is reported.

4. **Saving the Checkpoint**: Once the network is trained, the model is saved along with necessary information such as hyperparameters, class-to-index mapping, and optimizer state.

5. **Loading the Checkpoint**: A function is provided to load a saved checkpoint, allowing users to rebuild the trained model for inference or further training.

6. **Inference for Classification**: A function is implemented to make predictions using the trained model. Given an input image, the function returns the top K most probable classes along with their probabilities.

7. **Sanity Checking**: The predictions made by the model are visually checked for correctness. An input image along with its predicted probabilities for the top 5 classes is displayed using matplotlib.

## How to Use
To utilize the flower species classifier:

1. Ensure that all necessary dependencies are installed. This project primarily relies on PyTorch and torchvision.
2. Download the [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) and organize it into training, validation, and testing directories.
3. Follow the provided Python scripts or Jupyter notebook cells to execute each step of the project.
4. Train the classifier using the provided training data and evaluate its performance on the validation set.
5. Test the trained network on the unseen test data to assess its accuracy.
6. Save the trained model checkpoint for future use.
7. Utilize the inference function to make predictions on new images.
8. Verify the predictions using the sanity checking function.

Feel free to customize and extend the project according to your requirements, such as using different pre-trained models, adjusting hyperparameters, or incorporating additional functionalities.

# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.
