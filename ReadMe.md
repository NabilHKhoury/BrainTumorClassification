# Brain Tumor Classification Project

## Overview
This project leverages advanced machine-learning techniques to classify brain tumors from MRI scans. By utilizing Convolutional Neural Networks (CNNs) and Transfer Learning, we have created a system that can accurately distinguish between different types of brain tumors such as glioma, meningioma, and pituitary tumors. I collaborated on this difficult project with Cray Minor. Please contact either of us at nkhoury@ucsd.edu and csminor@ucsd.edu if you have any questions! 

## Dataset
The dataset comprises MRI scans labeled with the corresponding tumor type. These images have been preprocessed and augmented to improve model performance.

## Files Description
- `tumorClassification.py`: The main Python script that includes data loading, preprocessing, model building, training, and evaluation.
- `cnn_classification.png`: Visualization of the CNN model architecture used for tumor classification.
- `confusion_matrix_class_report.png`: Confusion matrix and classification report for the CNN model.
- `confusion_matrix_cnn.png`: Confusion matrix for the CNN model's predictions.
- `confusion_matrix_transfer_learning.png`: Confusion matrix for the predictions made using Transfer Learning.
- `loss_function.png`: Graph showing the training and validation loss over epochs for the models.
- `transfer_learning_classification.png`: Visualization of the Transfer Learning model architecture.
- `tumor_classifcation.png`: Sample output showing tumor classification results on the validation dataset.
- `tumor_classifcation_test_data.png`: Sample output showing tumor classification results on the test dataset.
- `tumor_piechart`: Pie chart representing the distribution of different tumor types in the dataset.

## Installation
To run this project, you will need to install the following Python packages:
- NumPy
- Pandas
- TensorFlow
- Keras
- Matplotlib
- scikit-learn

You can install these packages using `pip`:
```bash
pip install numpy pandas tensorflow keras matplotlib scikit-learn
Usage
Run the tumorClassification.py script to train the model and evaluate its performance. The script will output the confusion matrices and classification report to help assess the model's accuracy.

python tumorClassification.py
Results
The model achieves high accuracy in classifying brain tumors from MRI scans. The confusion matrices and loss function graphs provide insight into the model's performance.

Contributing
If you'd like to contribute to this project, please fork the repository and submit a pull request.

License
This project is released under the MIT License.

