# **HealthCoder - Alzheimer's Disease Classification**

<img align="right" height="200" alt="img" src="https://neurosciencenews.com/files/2023/03/alzheimers-ai-neurosicneces-public.jpg" padding="3px"  />


## Project Overview

HealthCoder is a project focused on the classification of Alzheimer's disease using advanced AI techniques and brain MRI (Magnetic Resonance Imaging) images. The goal of the project is to develop an accurate and robust model that can assist in the early detection and diagnosis of Alzheimer's disease.

## Objective

The objective of the project is to leverage AI techniques and machine learning algorithms to classify brain MRI images into different stages of Alzheimer's disease. By accurately identifying the disease's progression, the model can aid healthcare professionals in making timely diagnoses and developing appropriate treatment plans.

The specific objectives of the project are as follows:

1. Preprocess the MRI images to enhance their quality, remove noise, and standardize the data.

2. Develop and train deep learning models using TensorFlow and Keras to accurately classify MRI images into different stages of Alzheimer's disease. Optimize and fine-tune the models to achieve high classification accuracy, precision, recall, and other performance metrics.

3. Evaluate the trained models using appropriate evaluation metrics and compare their performance to identify the most effective model.

4. Visualize the results, including the MRI images, model predictions, and evaluation metrics, to facilitate interpretation and analysis. Create a comprehensive report summarizing the project, including the methodology, results, limitations, and potential areas for further improvement.

## Dataset: Alzheimer MRI Preprocessed Dataset

<img align="center" alt="img" src="https://user-images.githubusercontent.com/77446629/242084454-97acbe96-06cb-4eda-a236-2476db3f165a.png" padding="3px"  />

The project utilizes the [Alzheimer MRI Preprocessed Dataset](https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images) obtained from Kaggle. The dataset consists of 6400 preprocessed MRI images, resized to 128 x 128 pixels, representing different stages of Alzheimer's disease.

### Dataset Details

- Total Images: 6400
- Classes:
  - Class 1: Mild Demented (896 images)
  - Class 2: Moderate Demented (64 images)
  - Class 3: Non Demented (3200 images)
  - Class 4: Very Mild Demented (2240 images)


## Technologies Used

- TensorFlow: An open-source machine learning framework used for building and training deep learning models.
- Keras: A high-level neural networks API that runs on top of TensorFlow. It provides an intuitive interface for designing and training models.
- Pandas: A powerful data manipulation library used for data preprocessing and analysis.
- Matplotlib: A popular plotting library used for data visualization, including the visualization of MRI images and performance metrics.
- NumPy: A fundamental library for scientific computing in Python, used for numerical operations and array manipulation.
- Scikit-learn: A machine learning library that provides tools for data preprocessing, model evaluation, and performance metrics.

## Data Preprocessing and Augmentation

In the initial steps of the project, the dataset of Alzheimer's disease brain MRI images undergoes preprocessing and augmentation to enhance the data quality and increase the robustness of the model. The following steps are performed:

- Splitting the Dataset: The original dataset, obtained from Kaggle, is split into train, validation, and test sets.
- Image Preprocessing: The images are resized to 128 x 128 pixels.
- Data Augmentation: The training data is augmented using techniques such as rescaling, shearing, and zooming to increase its diversity and improve the model's ability to generalize.
- Data Normalization: The validation and test data are rescaled for normalization.
- Directory Setup: Directories are set up to specify the location of the split images for the train, validation, and test sets.
- ImageDataGenerators: The Keras `ImageDataGenerator` is used to generate batches of augmented images for the training set and normalized images for the validation and test sets.
- Class Mode: The class mode is set to 'categorical' to support multi-class classification.

These steps ensure that the dataset is properly prepared for training and evaluating deep learning models for the classification of Alzheimer's disease using brain MRI images.

Please refer to the code provided for more details on the implementation.

```python
import splitfolders
from keras.preprocessing.image import ImageDataGenerator
# Set the path of the directory containing the original images
input_folder = '/kaggle/input/alzheimer-mri-dataset/Dataset'
output_folder = '/kaggle/working/Splitted'
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1
# Split the images into train-validation-test sets
splitfolders.ratio(input_folder, output_folder, seed=42, ratio=(train_ratio, validation_ratio, test_ratio))
# Define the ImageDataGenerators for data augmentation and normalization
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
# Set the directories for train, validation, and test sets
train_dir = '/kaggle/working/Splitted/train'
validation_dir = '/kaggle/working/Splitted/val'
test_dir = '/kaggle/working/Splitted/test'
# Create generators for train, validation, and test sets
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(128, 128), shuffle=True, seed=SEED, batch_size=64, class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(128, 128), seed=SEED, shuffle=True, batch_size=64, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(128, 128), shuffle=True, seed=SEED, batch_size=64, class_mode='categorical')
```


## AI Models Used

The project incorporates the following AI models for Alzheimer's disease classification:

### 1. CNN Models

The project utilizes various CNN models for classification:

- Custom CNN architecture
- CNN (Convolutional Neural Network) [Implementation Notebook](https://github.com/SARIT42/alzheimers-detection/blob/main/CNN/alzheimer-detection-cnn%20(1).ipynb)

### 2. Transfer Learning Models

The project employs transfer learning using pre-trained models:


- VGG16: [Implementation Notebook](https://github.com/SARIT42/alzheimers-detection/blob/main/Transfer%20Learning/alzeihmer-vgg.ipynb)
- VGG19: [Implementation Notebook](https://github.com/SARIT42/alzheimers-detection/blob/main/Transfer%20Learning/alzeihmer-vgg.ipynb)
- ResNet: [Implementation Notebook](https://github.com/SARIT42/alzheimers-detection/blob/main/Transfer%20Learning/alzheimer-resnet50.ipynb)
- MobileNetV2: [Implementation Notebook](https://github.com/SARIT42/alzheimers-detection/blob/main/Transfer%20Learning/alzheimer-mobilenetv2.ipynb)
- InceptionV3: [Implementation Notebook](https://github.com/SARIT42/alzheimers-detection/blob/main/Transfer%20Learning/alzheimers-inceptionv3.ipynb)
- DenseNet169: [Implementation Notebook](https://github.com/SARIT42/alzheimers-detection/blob/main/Transfer%20Learning/alzheimer-densenet169.ipynb)
- EfficientNetb0: [Implementation Notebook](https://github.com/SARIT42/alzheimers-detection/blob/main/Transfer%20Learning/alzheimer-efficientnetb0.ipynb)



### 3. Machine Learning Models

The project includes traditional machine learning algorithms for classification:

- Logistic Regression
- SVM (Support Vector Machine)
- Random Forest
-  [Implementation Notebook](https://github.com/SARIT42/alzheimers-detection/blob/main/Machine%20Learning/alzheimer-pca-ml.ipynb)


### 4. Hybrid Deep Learning Models

The project implements hybrid deep learning models combining deep learning with other algorithms:

- Alzheimer-CNN-with-XGBoost-GNB-SVM: A hybrid model combining CNN with XGBoost, Gaussian Naive Bayes (GNB), and SVM algorithms. [Implementation Notebook](https://github.com/SARIT42/alzheimers-detection/blob/main/Hybrid%20Learning/alzeihmer-cnn-with-xgboost-gnb-svm.ipynb)

- Alzheimer-VGG-with-SVM-GNB-XGBoost: A hybrid model combining VGG16 with SVM, GNB, and XGBoost algorithms. [Implementation Notebook](https://github.com/SARIT42/alzheimers-detection/blob/main/Hybrid%20Learning/alzeihmer-vgg-with-svm-gnb-xgboost.ipynb)


## Future Scope and Limitations
- Larger and more diverse datasets: Acquiring larger and more diverse datasets can help improve the performance and generalizability of the CNN model.
- Multi-modal data fusion: Incorporating multiple imaging modalities, such as functional MRI (fMRI), positron emission tomography (PET), or cerebrospinal fluid (CSF) biomarkers, along with MRI data, can provide complementary information for more accurate prediction. 
- Longitudinal analysis: Alzheimer's disease is a progressive condition that evolves over time. Incorporating longitudinal data and analyzing disease progression can offer valuable insights into the temporal patterns and changes in brain structures.
- Integration with clinical data: Combining MRI data with clinical information, such as cognitive test scores, medical history, genetic data, or lifestyle factors, can lead to a more comprehensive and accurate prediction model. 


## References
- https://www.biorxiv.org/content/10.1101/070441v4.full.pdf
- https://www.proquest.com/openview/13f1237a0a57216debf783897664a462/1?pq-origsite=gscholar&cbl=4424409
- https://www.mdpi.com/2075-4418/12/8/1833
- https://www.sciencedirect.com/science/article/pii/S0010482521003310
- https://philpapers.org/archive/ALMCOA.pdf
- https://github.com/LinLLiu/AD




