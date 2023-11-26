import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import models

#################Testing with testset images#################
print("Testing start!")
img = load_img("/home/guest2/Simple_FER/test/new.jpg",target_size = (48,48))#,color_mode = "grayscale")
img = np.array(img)
print(img.shape)

print("Loading complete")



train_datagen = ImageDataGenerator(
    rescale = 1./255,   ## rescale or normalize the images pixels, by dividing them 255
    shear_range = 0.2,  ## angle for slant of image in degrees
    zoom_range = 0.2,   ## for zoom in or out
    horizontal_flip = True 
)
training_set = train_datagen.flow_from_directory(
    "/home/guest2/Simple_FER/input_blurring/train/",   ## give path of training set
    target_size=(48,48),      ## target_size of image in which you want
    batch_size=32,
    #color_mode = "grayscale",
    class_mode = 'categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    "/home/guest2/Simple_FER/input_blurring/test/",
    target_size = (48,48),
    batch_size = 32,
    #color_mode = "grayscale",
    class_mode = 'categorical'
)




training_set.class_indices
label_dict = ['angry', 'disgust', 'fear','happy', 'neutral', 'sad', 'surprise']

test_image = img_to_array(img)
test_image = np.expand_dims(test_image, axis = 0)

model = tf.keras.models.load_model("/model/path")
result = model.predict(test_image)
print(result)
result[0]

res = np.argmax(result[0])

print('predicted Label for that image is: {}'.format(label_dict[res]))

train_loss, train_acc = model.evaluate(training_set)
test_loss, test_acc   = model.evaluate(test_set)
print("final train accuracy = {:.2f} , validation accuracy = {:.2f}".format(train_acc*100, test_acc*100))