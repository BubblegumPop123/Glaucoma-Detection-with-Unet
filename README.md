# Glaucoma Detection Using U-Net

## Project Description

This project aims to develop an automated system for glaucoma detection using deep learning techniques, specifically a U-Net architecture. The system processes retinal images to segment the optic disc and cup, calculates the Cup-to-Disc Ratio (CDR), and classifies the severity of glaucoma based on the CDR values.

Glaucoma is a chronic eye disease that can lead to vision loss if not diagnosed and treated early. Accurate segmentation of the optic disc and cup in retinal images is crucial for reliable glaucoma detection. This project utilizes convolutional neural networks (CNNs) to achieve high accuracy in segmentation and classification tasks.

## Features

- **Data Loading and Preprocessing**: Efficiently loads and preprocesses retinal images and masks.
- **U-Net Model**: Constructs a U-Net architecture for precise segmentation of optic disc and cup.
- **Training with Callbacks**: Implements model training with early stopping and model checkpointing to save the best model.
- **Evaluation**: Visualizes segmentation results and calculates CDR to classify glaucoma severity.

## Provided Dataset

Alongside the project code, a dataset of retinal images and corresponding masks is provided. The dataset includes images of retinal scans and manually annotated masks for the optic disc and cup. The images are in `.png` format, and the masks are in `.tif` format, ensuring a clear distinction between the different regions of interest.

---

## Guide

### 1. Setup

#### Prerequisites

Ensure you have Python 3.x installed along with the necessary libraries: TensorFlow, OpenCV, NumPy, and Matplotlib. Google Colab can be used optionally for running the project on the cloud.

#### Installation

Install the required libraries using pip:

```bash
pip install tensorflow opencv-python numpy matplotlib
```

### 2. Loading Data

The data loading function efficiently loads retinal images and their corresponding masks from the provided dataset directory. It resizes the images to 256x256 pixels and normalizes them for further processing. The images are converted to RGB format, and the masks are prepared to distinguish between the optic disc and cup.

### 3. Building the U-Net Model

The U-Net model is constructed using the Keras library. The model architecture includes a contracting path with convolutional and max-pooling layers to capture context, followed by an expansive path with transposed convolutions for precise localization. Dropout layers are used to prevent overfitting.

### 4. Training the Model

The training process involves splitting the data into training and validation sets. Data augmentation techniques are applied to increase the diversity of the training data. The model is compiled with an appropriate optimizer and loss function. Training is performed with callbacks such as ModelCheckpoint to save the best model and EarlyStopping to prevent overfitting by stopping the training early if no improvement is observed.

### 5. Evaluating the Model

Post-training, the model's performance is evaluated by visualizing the predicted masks alongside the original images. The Cup-to-Disc Ratio (CDR) is calculated based on the segmented masks. The CDR values are then used to classify the severity of glaucoma into categories such as Normal, Moderate, and Severely Glaucomatous.

### 6. Visualization and Classification

The evaluation includes plotting the original images, predicted optic disc, and cup masks. The CDR is calculated by counting the white pixels in the predicted masks, and the classification of glaucoma severity is performed based on these CDR values. This process helps in understanding the model's accuracy and its practical application in glaucoma detection.

---

By following this guide, you can set up, train, and evaluate the U-Net model for glaucoma detection, gaining insights into medical image processing and the application of deep learning in healthcare.
