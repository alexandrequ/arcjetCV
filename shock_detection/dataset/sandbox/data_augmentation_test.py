from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import scipy.ndimage
from matplotlib import pyplot
import numpy as np
import cv2
# load the image
img = load_img('pika.png')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
height_shift_range=0.1,shear_range=0.15,
zoom_range=0.1,channel_shift_range = 10, horizontal_flip=True)


image_path = 'pika.png'

image = np.expand_dims(cv2.imread(image_path), 0)
save_here = 'train_frames_aug'

datagen.fit(image)

# prepare iterator
it = datagen.flow(samples, batch_size=1)

for x, val in zip(datagen.flow(image,                    #image we chose
        save_to_dir=save_here,     #this is where we figure out where to save
         save_prefix='img_aug',        # it will save the images as 'aug_0912' some number for every new augmented image
        save_format='png'),range(10)) :     # here we define a range because we want 10 augmented images otherwise it will keep looping forever I think
        print("hello")
