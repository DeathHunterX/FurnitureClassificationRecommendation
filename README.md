# Furniture Classification and Recommendation
## Objective in this project: 
Task 1: Classify images according to furniture category (beds; chairs; dressers; lamps; sofas; tables)

Task 2: Recommend 10 furniture items in our dataset which is similar to the input furniture item image from users. You are required to define a metric of “similarity” between two furniture items.

Task 3: The extension of the model in Task 2, the recommended furniture items must be in the same interior styles with the style of the input images. In order to fulfill this task, you are required to build a model to recognize the style of a furniture item.

## About Dataset
### Dataset Description:
- There are 1 dataset which is Furniture_Data in Furniture_Data zip file.
- These datasets contain 06 furniture categories and each category has 17 interior styles:
    - Furniture categories (90084 images toal):
        - beds  - 6578 images
        - chairs  - 22053 images
        - dressers - 7871 images
        - lamps  - 32402 images
        - sofas  - 4080 images 
        - tables - 17100 images
        - 
    - Interior styles of each furniture items:
        - Asian
        - Beach
        - Contemporary
        - Craftman
        - Eclectic
        - Farmhouse
        - Industrial
        - Mediterranean
        - Midcentury
        - Modern
        - Rustic
        - Scandinavian
        - Southwestern
        - Traditional
        - Transitional
        - Tropical
        - Victorian



## Data Source
Link of dataset: https://drive.google.com/file/d/1yF_pa86YN2O5pS0-TIQqYxRgk7z2fSVP/view?usp=sharing

## Technology/Infrastructure/Models
### Technology Stack
#### Programming Languages
- Python: Utilized for data analysis, manipulation, and visualization.
- 
#### Libraries/Frameworks
- **Pandas:** Pandas is a powerful data manipulation and analysis library for Python, essential for handling structured data efficiently.
- **NumPy:** NumPy is fundamental for numerical computing in Python, providing support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.
- **Matplotlib:** Matplotlib is a versatile plotting library that enables the creation of a wide variety of plots and visualizations, essential for data exploration and presentation.
- **TensorFlow:** TensorFlow is an open-source machine learning framework developed by Google, widely used for building and training deep learning models across a range of tasks.
- **Tensorflow - Keras:** Keras is a high-level neural networks API, built on top of TensorFlow, that provides a user-friendly interface for building and training deep learning models.
- **Sci-kit Learn:** Sci-kit learn is a simple and efficient tool for data mining and data analysis, offering various algorithms and tools for machine learning tasks such as classification, regression, clustering, and dimensionality reduction.

#### Machine Learning Models
- **Deep Learning Models:** Building convolutional neural networks (CNN) using TensorFlow and Keras for image classification tasks.
- **Model Evaluation:** I employ fine tuning for Adam and SGD optimizer as well as Early Stopping function in order to evaluate and improve the performance of my models.

### Model Architecture

### Training
The model is trained using the training dataset with the objective of minimizing a categorical cross-entropy loss function. During training, the model's weights are adjusted iteratively using the Adam optimizer.

#### Training Parameters
- For Self-build model
    - Batch Size: 32
    - Image Size: 224 x 224
    - Optimizer: Adam
    - Number of Epochs: 30
    - Learning Rate: 0.001

- For AlexNet model
    - Batch Size: 32
    - Image size: 227 x 227
    - Optimizer: SGB
    - Learning Rate: 0.1
    - Number of Epochs: 30 

### How to use
#### 1. Clone this repository

#### 2. Install necessary libraries/frameworks

```bash
    pandas
    numpy
    matplotlib
    tensorflow
    opencv-python
    opencv-contrib-python
    scikit-learn
    nomkl # to handle error. Detailed: https://github.com/dmlc/xgboost/issues/1715
    
```
