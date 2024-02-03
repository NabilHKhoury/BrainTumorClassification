from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import keras
from keras.applications.vgg16 import preprocess_input,VGG16
import tensorflow as tf
from keras.models import Sequential,Model
from keras.layers import MaxPooling2D,Conv2D,Dense,BatchNormalization,Dropout,GlobalAveragePooling2D,Flatten,Input
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.metrics import classification_report
from keras.utils.vis_utils import plot_model
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# **This function converts the image sets to a dataframe.**

# %%
# Function to load images
def convert_image_to_dataset(file_location):
    label=0
    df=pd.DataFrame()
    for category in glob(file_location+'/*'):
        for file in tqdm(glob(category+'/*')):
            img_array=cv2.imread(file)
            img_array=cv2.resize(img_array,(224, 224))
            data=pd.DataFrame({'image':[img_array],'label':[label]})
            df=df.append(data)
        label+=1
    return df.sample(frac=1).reset_index(drop=True)

# %% [markdown]
# **This function converted the output labels of each tumor type to their respective catergories.**

# %%
# Function to convert output labels to its class of tumor.
def inverse_classes(num):
    if num==0:
        return 'Glioma Tumor'
    elif num==1:
        return 'Meningioma Tumor'
    elif num==2:
        return 'No Tumor'
    else:
        return 'Pituitary Tumor'
    

# %% [markdown]
# **Conversion of images to train and test sets.**

# %%
train_data=convert_image_to_dataset(r'C:\Users\Nabil\GitHub\neural_ds_proj\new_data\archive\Training')

train_x=np.array(train_data.image.to_list())


# %%
test_data=convert_image_to_dataset(r'C:\Users\Nabil\GitHub\neural_ds_proj\new_data\archive\Testing')
test_x=np.array(test_data.image.to_list())

# %% [markdown]
# <h1 style="background-color:#C0C0C0;font-family:newtimeroman;font-size:550%;text-align:center;border-radius: 15px 10px;padding: 5px"><b>Visualizing Raw Dataset</b></h1>
# 

# %% [markdown]
# **This pie chart shows a break down of the tumors by the percentage of each in the dataset.**

# %%
plt.pie(train_data.label.value_counts(),startangle=90,explode=[0.1,0.1,0.1,0.2],autopct='%0.2f%%',
        labels=['Meningioma_tumor', 'Pituitary Tumor', 'No Tumor', 'Glioma Tumor'],radius=3)
plt.show()
plt.savefig('tumor_piechart.png', dpi=200,format='png', bbox_inches='tight')


# %%
plt.pie(test_data.label.value_counts(),startangle=90,explode=[0.1,0.1,0.1,0.2],autopct='%0.2f%%',
        labels=['Meningioma_tumor', 'Pituitary Tumor', 'No Tumor', 'Glioma Tumor'],radius=3)
plt.show()
plt.savefig('tumor_piechart_test_data.png', dpi=200,format='png', bbox_inches='tight')


# %% [markdown]
# **Here we display a random subset of the data to show the actual MRI scans that our model is being trained on.**
# 

# %%
plt.figure(figsize=(20,15))
for i in range(12):
    plt.subplot(4,3,(i%12)+1)
    index=np.random.randint(2000)
    plt.title('This a {0}'.format(inverse_classes(train_data.label[index])),fontdict={'size':20,'weight':'bold'})
    plt.imshow(train_data.image[index])
    plt.tight_layout()
    plt.savefig('tumor_classifcation.png', dpi=200,format='png', bbox_inches='tight')


# %%
plt.figure(figsize=(20,15))
for i in range(12):
    plt.subplot(4,3,(i%12)+1)
    index=np.random.randint(200)
    plt.title('This a {0}'.format(inverse_classes(test_data.label[index])),fontdict={'size':20,'weight':'bold'})
    plt.imshow(test_data.image[index])
    plt.tight_layout()
    plt.savefig('tumor_classifcation_test_data.png', dpi=200,format='png', bbox_inches='tight')


# %% [markdown]
# <h1 style="background-color:#C0C0C0;font-family:newtimeroman;font-size:550%;text-align:center;border-radius: 15px 10px;padding: 5px"><b>Callbacks Functions</b></h1>
# 

# %%
early_stop=EarlyStopping(patience=3)
reduceLR=ReduceLROnPlateau(patience=2)

# %% [markdown]
# **If there is a GPU available first try to train on that so it doesn't take as long.**

# %%
import tensorflow as tf

# check if GPU is available, otherwise use CPU
if tf.config.list_physical_devices('GPU'):
    device_name = '/GPU:0'
else:
    device_name = '/CPU:0'

# instantiate a distribution strategy
strategy = tf.distribute.OneDeviceStrategy(device=device_name)


with strategy.scope():
    model_cnn=Sequential()
    model_cnn.add(Input(shape=(224,224,3)))
    # model_cnn.add(Conv2D(128,(3,3)dilation_rate=2)) is the dialation layer

    model_cnn.add(Conv2D(128,(3,3), dilation_rate = 2))
    model_cnn.add(MaxPooling2D((2,2)))
    model_cnn.add(BatchNormalization())
    model_cnn.add(Conv2D(64,(3,3)))
    model_cnn.add(MaxPooling2D((2,2)))
    model_cnn.add(BatchNormalization())
    model_cnn.add(Conv2D(32,(3,3)))
    model_cnn.add(MaxPooling2D((2,2)))
    model_cnn.add(BatchNormalization())
    model_cnn.add(Flatten())
    model_cnn.add(Dense(128,activation='relu'))
    model_cnn.add(Dropout(0.2))
    model_cnn.add(Dense(64,activation='relu'))
    model_cnn.add(Dense(4,activation='softmax'))
    model_cnn.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# %%
r1=model_cnn.fit(train_x,train_data.label,validation_split=0.1,epochs=20,callbacks=[reduceLR])

# %%
plt.figure(figsize=(10,8))
plt.plot(r1.history['val_accuracy'])
plt.plot(r1.history['accuracy'])
plt.legend(['val_accuracy','accuracy'])
plt.show()
plt.savefig('accuracy.png', dpi=200,format='png', bbox_inches='tight')



# %%
plt.figure(figsize=(10,8))
plt.plot(r1.history['val_loss'])
plt.plot(r1.history['loss'])
plt.legend(['val_loss','loss'])
plt.show()
plt.savefig('loss_function.png', dpi=200,format='png', bbox_inches='tight')


# %%
test_loss, test_accuracy = model_cnn.evaluate(test_x,test_data.label)

# Predictions on Test Datasets using CNN model
pred =model_cnn.predict(test_x)
test_pred=np.argmax(pred,axis=1)
print(classification_report(test_data.label,test_pred))

# **This function creates a bar plot that breaks down the precision, recall, and f1 score for each tumor class on the test dataset.**

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def plot_classification_report(y_true, y_pred, class_labels):
    """
    Plots a bar plot of the classification report.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    class_labels (list): List of class labels.

    Returns:
    None
    """
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)

    precision = []
    recall = []
    f1_score = []
    support = []
    for label in class_labels:
        precision.append(report[label]['precision'])
        recall.append(report[label]['recall'])
        f1_score.append(report[label]['f1-score'])
        support.append(report[label]['support'])

    x = range(len(class_labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x, precision, width, label='Precision')
    ax.bar([i + width for i in x], recall, width, label='Recall')
    ax.bar([i + width*2 for i in x], f1_score, width, label='F1-Score')

    ax.set_xticks([i + width for i in x])
    ax.set_xticklabels(class_labels)
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Classification Report')
    ax.legend()

    plt.show()


plt.figure(figsize=(15,12))
for i in range(4):
    plt.subplot(3,2,(i%12)+1)
    index=np.random.randint(200)
    pred_class=inverse_classes(np.argmax(model_cnn.predict(np.reshape(test_x[index],(-1,224,224,3))),axis=1))
    plt.title('This is a {0}, and it is predicted as {1}'.format(inverse_classes(test_data.label[index]),pred_class),
              fontdict={'size':15})
    plt.imshow(test_x[index])
    plt.tight_layout()
    plt.savefig('cnn_classification.png', dpi=200,format='png', bbox_inches='tight')


class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor','pituitary_tumor']
report =classification_report(test_data.label,test_pred)
plot_classification_report(test_data.label,test_pred, class_labels)
plt.savefig('confusion_matrix_cnn.png', dpi=200,format='png', bbox_inches='tight')
plt.close()


import tensorflow as tf
from keras.applications.vgg16 import VGG16

# check if GPU is available, otherwise use CPU
if tf.config.list_physical_devices('GPU'):
    device_name = '/GPU:0'
else:
    device_name = '/CPU:0'

# instantiate a distribution strategy
strategy = tf.distribute.OneDeviceStrategy(device=device_name)


with strategy.scope():
    vgg_model = VGG16(weights='imagenet',include_top=False)
    for layers in vgg_model.layers:
        layers.trainable=False
    x=vgg_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(128,activation='relu')(x)
    x=Dropout(0.15)(x)
    output=Dense(4,activation='softmax')(x)
    model2=Model(inputs=vgg_model.input,outputs=output)
    model2.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# %%
transfer_learning=model2.fit(train_x,train_data.label,validation_split=0.1,epochs=20,callbacks=[early_stop,reduceLR])

# %%
test_loss, test_accuracy = model2.evaluate(test_x,test_data.label)
# **After 20 training epochs we can see that this network performs similar and ends over 98% on the training data, and at 93% on the validation set. Below we plotted the model accuracy and validation set accuracy over its epochs.** 

# %%
plt.figure(figsize=(10,8))
plt.plot(transfer_learning.history['val_accuracy'])
plt.plot(transfer_learning.history['accuracy'])
plt.legend(['val_accuracy','accuracy'])
plt.show()

# %% [markdown]
# # Loss Function Visualization

# %% [markdown]
# **This vizualization is the value we lost over the epochs, we can see that our model's loss value descreases over the training approaching 0.2 on the validation accuracy meaning it still performed well on non-labeled images**

# %%
plt.figure(figsize=(10,8))
plt.plot(transfer_learning.history['val_loss'])
plt.plot(transfer_learning.history['loss'])
plt.legend(['val_loss','loss'])
plt.show()

# %%
test_pred_transfer=np.argmax(model2.predict(test_x),axis=1)
print(classification_report(test_data.label,test_pred_transfer))

# %% [markdown]
# **This is a classification report from the transfer learning model, we can see that it again performed worst on the glioma tumor class because it has less training data.**

# %%
class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor','pituitary_tumor']
report =classification_report(test_data.label,test_pred)
plot_classification_report(test_data.label,test_pred_transfer, class_labels)
plt.savefig('confusion_matrix_transfer_learning.png', dpi=200,format='png', bbox_inches='tight')
plt.close()

# %% [markdown]
# **This shows the ground truth labels of the images and what the transfer learning classification model predicted the label to be.**
# 

# %%
plt.figure(figsize=(15,12))
for i in range(4):
    plt.subplot(3,2,(i%12)+1)
    index=np.random.randint(200)
    pred_class=inverse_classes(np.argmax(model2.predict(np.reshape(test_x[index],(-1,224,224,3))),axis=1))
    plt.title('This image is of {0} and is predicted as {1}'.format(inverse_classes(test_data.label[index]),pred_class),
              fontdict={'size':15})
    plt.imshow(test_x[index])
    plt.tight_layout()
    plt.savefig('transfer_learning_classification.png', dpi=200,format='png', bbox_inches='tight')


# %% [markdown]
# **This is another sample set of images with the ground truth labels and what the convoluted neural network predicted the label to be, to compare both models.**

# %%
plt.figure(figsize=(15,12))
for i in range(4):
    plt.subplot(3,2,(i%12)+1)
    index=np.random.randint(200)
    pred_class=inverse_classes(np.argmax(model_cnn.predict(np.reshape(test_x[index],(-1,224,224,3))),axis=1))
    plt.title('This image is of {0} and is predicted as {1}'.format(inverse_classes(test_data.label[index]),pred_class),
              fontdict={'size':15})
    plt.imshow(test_x[index])
    plt.tight_layout()


