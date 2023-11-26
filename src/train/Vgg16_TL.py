import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential

from keras.applications import VGG16
from keras.layers import Flatten, Dense

from keras.callbacks import EarlyStopping



#################preprocessing the training dataset#################

train_datagen = ImageDataGenerator(
    rescale = 1./255,   ## rescale or normalize the images pixels, by dividing them 255
    shear_range = 0.2,  ## angle for slant of image in degrees
    zoom_range = 0.2,   ## for zoom in or out
    horizontal_flip = True 
)
training_set = train_datagen.flow_from_directory(
    "/your/training data/path",   ## give path of training set
    target_size=(48,48),      ## target_size of image in which you want
    #target_size=(128,128)
    #target_size=(256,256)
    batch_size=32,
    class_mode = 'categorical'
)

#################preprocessing the testing dataset#################

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    "/home/guest2/Simple_FER/input_blurring/test/",
    target_size = (48,48),
    #target_size=(128,128)
    #target_size=(256,256)
    batch_size = 32,
    class_mode = 'categorical'
)

#################Building the CNN Model#################

#base model#
base_model = tf.keras.applications.VGG16(input_shape=(48,48,3) ,include_top=False, weights="imagenet")
base_model.summary()

#Freezing Layers#
for layer in base_model.layers[:15]:
    layer.trainable=False
#base_model.trainable = False :training only dense layer

print("Loading Base Model complete")    


model = Sequential()
model.add(base_model)
## add output layer
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(7, activation='softmax'))


#################Compiling the CNN model#################


model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


#################Training the model#################
early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
history = model.fit(training_set, 
                    epochs=50,
                    batch_size=32,
                    validation_data = test_set,
                    shuffle=True,
                    callbacks = [early_stop]
                   )
                   
                   
#################Save the trained model#################  
model.save('Vgg16_TL.h5')  


#################Plot the training results################# 

#plot accuracy
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('accuracy.png')

#plot loss
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('loss.png')

