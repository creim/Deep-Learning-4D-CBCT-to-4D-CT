from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

train_dir = '/home/creim/Desktop/TrainModel/Train Data 4D/'
validation_dir = '/home/creim/Desktop/TrainModel/Validation Data 4D/'

train_CT_dir = train_dir + '4D CT Data'
train_CBCT_dir = train_dir + 'Registrated 4D CBCT Bin 0'

validation_CT_dir = validation_dir + '4D CT'
validation_CBCT_dir = validation_dir + '4D Registrated CBCT'

num_CT_tr = len(os.listdir(train_CT_dir))
num_CBCT_tr = len(os.listdir(train_CBCT_dir))
num_CT_val = len(os.listdir(train_CT_dir))
num_CBCT_val = len(os.listdir(train_CBCT_dir))

total_train = num_CT_tr + num_CBCT_tr
total_val = num_CT_val + num_CBCT_val

batch_size = 10
epochs = 15

#CT images have 512x512 Pixel, are cropped by a little
IMG_HEIGHT = 512
IMG_WIDTH = 512

#Generator for our training data
#Applying online augmentations while feeding the data into the cnn:
'''Augmentations: -random rotation at 45 degrees
                  -width-shift .15
                  -height-shift .15
                  -horizontal-flip
                  -zoom'''
train_image_generator = ImageDataGenerator(rotation_range=45,
                                           width_shift_range=.15,
                                           height_shift_range=.15,
                                           horizontal_flip=True,
                                           zoom_range=0.5
                                           )

#Generator for validation data, randomly flip horizontal and rotate
validation_image_generator = ImageDataGenerator(horizontal_flip=True, )

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              color_mode='grayscale',
                                                              class_mode='binary')

#Visualize Training images
#sample_training_images, _ = next(train_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        img = tf.squeeze(img, axis=2)
        ax.imshow(img, cmap='gray')
        ax.axis('on')
    plt.tight_layout()
    plt.show()

plotImages(augmented_images)



#Create the Model - similar architecture to VGG19
'''1 Conv2D Layers(512,512,64) 3x3Conv, Pooling Layer(256,256,64), 
2 Conv2D Layers(256,256,128)3x3Conv, Pooling Layer(128,128,128),
2 Conv2D Layers(128,128,256)3x3Conv, Pooling Layer(64,64,256),
2 Conv2D Layers(64,64,512)3x3Conv, Pooling Layer(32,32,512),
Flatten(131072x1), Dense(512x1), Dense(1x1)'''

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH,1)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(128, 3, padding='same', activation ='relu'),
    Conv2D(128, 3, padding='same', activation ='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
#Takes obviously longer to train than the one before, but seems
#to need more data to increase accuracy!!
    
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#Train the model
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    verbose = 1,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)


#Visualize Training Results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
print('Showing the training results:')
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
fig = plt.gcf()
fig.savefig('Training and validation Accuracy.png')


plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
fig1 = plt.gcf()
fig1.savefig('Training and validation Loss.png')
plt.show()



model.save('CT-CBCT-classifier-4D-Data-Augmentation/Registration.h5')

