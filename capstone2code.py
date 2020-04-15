# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 06:58:43 2020

@author: feder
"""

##### Import packages
import os
from PIL import Image
import pandas as pd
import random 
import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import pickle 

#### Import data
train_directory = './train'
test_directory = './test'
validation_directory = './val'

#### Set seed
random.seed(123456789)


## Show picture as example of unresized one
# ![title](/notebooks/ImageTest/image.png "ShowMyImage")

## Images are all different sizes, so resize them to size of smallest
## Images have to be of the same size because of the way CNNs work



# Get sizes of all images


sizes = []
for root, dirs, files in os.walk(".", topdown=True):
    for name in files:
        rootname = os.path.join(root, name)
        if rootname.lower().endswith(('.jpg', '.jpeg')):
            im = Image.open(rootname)
            sizes.append(im.size)
            im.close()

# Convert list of tuples into dataframe
sizes = pd.DataFrame(sizes, columns =['x', 'y']) 

# find smallest
print(min(sizes.x)) # 384
print(min(sizes.y)) # 127    
        
# Resize all to smallest size
imsize = (384,127) 
im_width = 384
im_height = 127



## Run this only once
#################################################################
## Resizer function
## Even if pictures are all grayscale, some are formatted as rgb with identical R,G,B
## convert to grayscale on image opening
## WARNING: This overwrites the files so back them up
def resizer(dirname):
    for file in os.listdir(dirname):
        im = Image.open(os.path.join(dirname, file)).convert('L')
        im_resized = im.resize(imsize, Image.ANTIALIAS)
        im_out = os.path.join(dirname, file)
        im_resized.save(im_out, "JPEG", quality = 95)
        
## Ran this for each dirname/NORMAL and dirname/PNEUMONIA
resizer('./train/NORMAL')
resizer('./train/PNEUMONIA')

resizer('./test/NORMAL')
resizer('./test/PNEUMONIA')

resizer('./val/NORMAL')
resizer('./val/PNEUMONIA')
#################################################################


## Show picture as example of resized one 
# ![title](/notebooks/ImageTest/image.png "ShowMyImage")


## Now that the data is all in the right format, look at it
## Collect intensity of every pixel in every image
## Only do this once
intensities = []
for root, dirs, files in os.walk(".", topdown=True):
    for name in files:
        rootname = os.path.join(root, name)
        im = Image.open(rootname)
        if im.size == imsize:
            intensities.extend(  list( im.getdata() )     )
        im.close()
            
## Save Intensities to a file because of how big it is to avoid having to rebuild it each time                                          
intensities_df = pd.DataFrame(data={"Intensity": intensities})
intensities_df.to_csv("./intensities.csv", sep=',',index=False)
        
## Too much data to reasonably plot, instead take a random sampling (0.0001%).
# A large sample taken completely randomly should be representative
intensities_sample = random.sample(intensities, round(0.0001 * len(intensities))  )
intensities_sample = [x for x in intensities_sample if type(x) is int] # Remove stray tuples 
## Save sample to file also
sample_df = pd.DataFrame(data={"Intensity": intensities_sample})
sample_df.to_csv("./intensities_sample.csv", sep=',',index=False)

# Remove intensities and _df to free up memory
del intensities
del intensities_df




## Code to retrieve sample
intensities_sample = pd.read_csv('intensities_sample.csv')
# intensities_sample = intensities_sample.sample(n= round(0.05 * len(intensities_sample)) )
intensities_sample = intensities_sample.Intensity
intensities_sample = [x for x in intensities_sample if type(x) is int] # Remove stray tuples 

## plot histogram of distribution of pixel intensities
plt.hist(intensities_sample, 100, facecolor='red')
plt.title('Histogram of Distribution of Pixel Intensities')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


##
## Comment on distribution
## Apart from the large proportion at 0 (black pixels), there is a relatively
## even distribution of intensities with few completely white pixels. This fits
## from what we can see in the example images above.
'''



'''

#### Look at data
# Take in pixel types
# plot histogram of distribution of pixel intensities
# find out avg. pixel intenstiy, std deviation of pixel intensity, %tiles, quartiles these not a priority




### Use convolutional neural network

# start with 2-3 layer convolution netowrk
# convolution/poolinglayers
#check performance

## 


  
n_train_samples = 5216 # 1341 Normal + 3875 Pneumonia 
n_validation_samples = 16 # 8 Normal + 8 Pneumonia 
n_test_samples = 624 # 234 Normal + 390 Pneumonia 
epochs = 10
batch_size = 16


'''
if K.image_data_format() == 'channels_first': 
    input_shape = (3, img_width, img_height) 
else: 
    input_shape = (img_width, img_height, 3) 
'''


### Build cnn
model = Sequential() 

model.add(Conv2D(32, (2, 2), strides = (1,1), input_shape = (384,127,3) ) ) 
#model.add(MaxPooling2D(pool_size =(3, 3))) 

model.add(Flatten())
model.add(Dense(32, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu')) 
#model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',  metrics=['accuracy'])




### Image augmentation
train_imd = ImageDataGenerator(rescale = 1. / 255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True) 
test_imd = ImageDataGenerator(rescale = 1. / 255) 



### Set flows from directories and run
train_generator = train_imd.flow_from_directory(train_directory, target_size =(im_width, im_height), batch_size = batch_size, class_mode ='binary') 
  
validation_generator = test_imd.flow_from_directory(validation_directory, target_size =(im_width, im_height), batch_size = batch_size, class_mode ='binary') 


## Set up checkpointing so that weights from every epoch are saved
checkpoint_path = "cnn_checkpoint_epoch_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath = checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')
callbacks_checkpoint = [checkpoint]
 
model_fitted = model.fit_generator(train_generator, steps_per_epoch = n_train_samples // batch_size, epochs = epochs, validation_data = validation_generator, validation_steps = n_validation_samples // batch_size, callbacks = callbacks_checkpoint) 



#model.save_weights('cnn.h5') 


with open('traininghistory', 'wb') as file_pi:
        pickle.dump(model_fitted.history, file_pi)



#########################


## Look at model performance and consider changes
# Plot learning curves

accuracy = model_fitted.history['accuracy']
val_accuracy = model_fitted.history['val_accuracy']
loss = model_fitted.history['loss']
val_loss = model_fitted.history['val_loss']

epoch_range = range(1, len(accuracy)+1)

plt.plot(epoch_range, accuracy, label='Training Accuracy')
plt.plot(epoch_range, val_accuracy, label='Validation Accuracy')
plt.title('Accuracies Across Epochs')
plt.legend()
plt.figure()


plt.plot(epoch_range, loss,  label='Training loss')
plt.plot(epoch_range, val_loss, label='Validation loss')
plt.title('Losses Across Epochs')
plt.legend()
plt.show()


model_history = pd.read_pickle(r'traininghistory')
## convert dict to dataframe
model_history_df = pd.DataFrame.from_dict(model_history)
print(model_history_df)
## Best (accuracy and val_accuracy are both high and similar to each other) epoch is 4 so use that one
model = load_model('cnn_checkpoint_epoch_04.hdf5')
model.summary()



## Run test data through model
test_generator = test_imd.flow_from_directory(test_directory, target_size = (im_width,im_height), batch_size = batch_size, class_mode ='binary')
preds =model.evaluate(test_generator, verbose = 1)
test_loss = preds[0]
test_accuracy = preds[1]

print('When applied to the test set, the model has an accuracy of', round(test_accuracy, 2) ,'%')
