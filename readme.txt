READ ME

MODEL ACCURACY:
MODEL 1 TEST ACCURACY:94.05%
MODEL 1 VALIDATION ACCURACY: 93.83%
MODEL 1 BETA TEST ACCURACY: 97.91%
MODEL 1 BETA VALIDATION ACCURACY: 98.52%

MODEL 2 TEST ACCURACY:98.93%
MODEL 2 VALIDATION ACCURACY: 98.91%
MODEL 2 BETA TEST ACCURACY: 98.86 %
MODEL 2 BETA VALIDATION ACCURACY:98.52%

MODEL 3 TEST ACCURACY: 99.08%
MODEL 3 VALIDATION ACCURACY: 98.91%
MODEL 3 BETA TEST ACCURACY: 99.31 %
MODEL 3 BETA VALIDATION ACCURACY:98.52%

This python notebook should be ran in an environemt sutiable for .ipynb files such as Google Colab, Kaggle, Jupyter Notebook, and even Visual Studio.
Use over other IDEs is not advised.

To generate python equivelent functions and objects we must first install the proper libraries:
-pip install tensorflow
-pip install keras
-pip install matplotlib
-pip install numpy
-pip install sklearn
-pip install Ipython
-pip install shutil
-pip install itertools
-pipinstall imutils
-pip install plotly
-pip install seaborn

Once you have installed the appropriate backages you are one step to closer to using the neural network code!

LINK TO DATASET **** LINK TO DATASET **** LINK TO DATASET
https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset
Place the dataset in a selected path on your computer. It should be pre split into training and testing data, Place both those folders within one folder.
***********************************************************

Open BrainTumorClassificationCNN.ipynb in the proper enviorment (data-analysis friendly environment)


Throughout various pointed in the code you will see a the training path:C:user/path/.../Training 
as well as the testing data path: C:user/path/.../Testing.

In order to properly run the code you must locate the data set on your computer and set the paths to match for training and testing data respectively.
Some variables that will have this path associated with them include:
data_dir = "this is the training path" = C:user/path/.../Testing
test_dir = C:user/path/.../Testing
test = C:user/path/.../Testing
train = C:user/path/.../Training 

To ensure ALL of the paths have been correctly specified it is recommended to copy+ctrl+f+paste the path to locate the various instances in which it is used.

Once you have correctly ensure the packages are installed, ensure that you are able to import all the below libraries.

Now that the proper packages it is essential to import them and their needed functions
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential #Model initializer
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout #Regularization Methods and Layer types
from tensorflow.keras.layers import BatchNormalization #BatchNorm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD #optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint #verification callbacks

import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from tqdm import tqdm
import shutil
import itertools
import imutils
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools


Execution of this code should be done cell by cell to ensure proper understanding, the code provides comments to help user make necassary modifications.

If the user desires to change the number of convultional layers 
initial design: modelEXbeta.add(Conv2D(64,(7,7), input_shape=(64, 64, 1), padding='same', activation='relu'))
modified design: modelEXbeta.add(Conv2D(32,(7,7), input_shape=(64, 64, 1), padding='same', activation='relu'))

If the user wishes to do disbale zero padding
initial design: -modelEXbeta.add(Conv2D(64,(7,7), input_shape=(64, 64, 1), padding='same', activation='relu'))
modified design: -modelEXbeta.add(Conv2D(64,(7,7), input_shape=(64, 64, 1), padding='valid', activation='relu'))

If the user wishes to do change the kernel (filter) size
initial design: -modelEXbeta.add(Conv2D(64,(7,7), input_shape=(64, 64, 1), padding='same', activation='relu'))
modified design: -modelEXbeta.add(Conv2D(64,(3,3), input_shape=(64, 64, 1), padding='same', activation='relu'))

If the user wishes to change the actiavtion function
initial design: -modelEXbeta.add(Conv2D(64,(7,7), input_shape=(64, 64, 1), padding='same', activation='relu'))
modified design: -modelEXbeta.add(Conv2D(64,(3,3), input_shape=(64, 64, 1), padding='same', activation='softmax'))

####REFERENCES##### ####REFERENCES##### ####REFERENCES#####

UNDERSTANDING HOW BRAIN TUMORS ARE ANALYZED WITH MRI SCANS
https://www.dana-farber.org/brain-tumors/diagnosis/

LINK TO DATASET
https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset

LINK TO CODE THAT PROVIDED MODEL INSPIRATION https://www.kaggle.com/stpeteishii/brain-tumor-mri-torch-conv2d 
While the exact models were not used it is fair to give credit to a source that sparked ideas.

LINK TO CODE THAT HELPED NAVIGATE KERAS IMAGE AUGMENTATION/PROCESSING: https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720 
This website provided a clear understanding of the create certain augmentations. In combination with the overall source of inspiration for this code.

Book for Tensorflow-Keras that aided in the overall understanding of how tensorflow-keras works and how to optimize models to the best of our abilitiy:
Géron Aurélien. (2019). Deep Computer Vision Using Convolutional Neural Networks. In Hands-on machine learning with scikit-learn and tensorflow concepts, tools, and techniques to build Intelligent Systems (pp. 431–437). essay, O'Reilly. 
