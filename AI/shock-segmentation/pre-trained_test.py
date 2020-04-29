
import os
from keras_segmentation.models.unet import vgg_unet

dataset_path = os.getcwd() + "/dataset"

model = vgg_unet(n_classes=51 ,  input_height=672, input_width=1216 )

################################################################


model.train(
    train_images =  dataset_path + "/images_train/",
    train_annotations = dataset_path + "/masks_train/",
    checkpoints_path = dataset_path + "/checkpoints/vgg_unet_1" , epochs=5
)


# load any of the 3 pretrained models

out = model.predict_segmentation(
    inp="pika.png",
    out_fname="out.png"
)
