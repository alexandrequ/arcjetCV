# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras_segmentation.models.unet import vgg_unet
import numpy as np
import argparse


model = vgg_unet(n_classes=51 ,  input_height=128, input_width=128 )

# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
#images = "train_frames"
#masks = "train_masks"
#image_datagen.fit(images, augment=True, seed=seed)
#mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    'train_frames/',
    target_size=(124, 124),
    save_to_dir="train_frames_aug",
    color_mode='rgb',
    batch_size=150,
    save_format='png',
    save_prefix='img_aug',
    seed=seed,
    subset='training')

mask_generator = mask_datagen.flow_from_directory(
    'train_masks/',
    target_size=(124, 124),
    save_to_dir="train_masks_aug",
    save_prefix='img_aug',
    batch_size=150,
    save_format='png',
    seed=seed,
    subset='training')

save_here = 'train_frames_aug'

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

for x, val in train_generator:     # here we define a range because we want 10 augmented images otherwise it will keep looping forever I think
    break
