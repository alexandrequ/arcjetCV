from keras.models import Input,load_model
from keras.layers import Dropout,concatenate,UpSampling2D
from keras.layers import Conv2D, MaxPooling2D

from keras_segmentation.predict import predict_multiple
from keras_segmentation.train import find_latest_checkpoint

from glob import glob
from classes.Functions import splitfn

TRAIN = 0
LOAD = 1
APPLY = 1
input_height,input_width = 128,128
n_classes = 3
epochs= 2
ckpath = "checkpoints/mynet_arcjetCV"

##############################################################################
img_input = Input(shape=(input_height,input_width , 3 ))

conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
conv1 = Dropout(0.2)(conv1)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D((2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Dropout(0.2)(conv2)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D((2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Dropout(0.2)(conv3)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=-1)
conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
conv4 = Dropout(0.2)(conv4)
conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1)
conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
conv5 = Dropout(0.2)(conv5)
conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

out = Conv2D( n_classes, (1, 1) , padding='same')(conv5)
##############################################################################

from keras_segmentation.models.model_utils import get_segmentation_model
model = get_segmentation_model(img_input ,  out ) # this would build the segmentation model

if LOAD:
    latest_weights = find_latest_checkpoint(ckpath)
    model.load_weights(latest_weights)

if TRAIN:
    print("STARTED TRAINING\n")
    model.train( 
        train_images =  "./train_frames/",
        train_annotations = "./train_masks/",
        checkpoints_path = ckpath , epochs=epochs,
        batch_size=32
    )
    print('FINISHED TRAINING')

##out = model.predict_segmentation(
##    inp="./val_frames/ms_8ply_0000_0_10.png",
##    out_fname="test_output.png"
##)

### Apply network to target imgs
if APPLY:
    inp_dir="./validation/"
    out_dir="./outputs/"
    regex = inp_dir + "frame_????.png"
    imgpaths = sorted(glob(regex))
    
    for p in imgpaths:
        folder,name,ext = splitfn(p)
        out = model.predict_segmentation(
            inp=p,
            out_fname=out_dir+name+ext,
            colors=[(0,0,255),(0,255,0),(255,0,0)]
        )
        lname = name.split('_')
        if lname[-1]=='0' and lname[-2]=='0':
            print(name)
