# import Python libraries
import cv2 as cv
import numpy as np
from glob import glob
import os

# import ML modules
from keras.models import Input
from keras.layers import Dropout,concatenate,UpSampling2D
from keras.layers import Conv2D, MaxPooling2D
from keras_segmentation.train import find_latest_checkpoint, train
from keras_segmentation.models.model_utils import get_segmentation_model
from keras.utils import plot_model

def get_unet_model(img, nclasses=3, ckpath = "ML/checkpoints_mosaic/mynet_arcjetCV"):
    """Get custom neural net model for model/shock/background segmentation

    Args:
        img (2D np array): opencv image
        nclasses (int, optional): Number of classes. Defaults to 3.
        ckpath (str, optional): Checkpoints path for saving model weights. Defaults to "ML/checkpoints_mosaic/mynet_arcjetCV".

    Returns:
        model: keras model object
    """
    cv.imwrite("frame.png", img)

    height = img.shape[0]
    width= img.shape[1]
    hpx = height- height%4
    wpx = width- width%4

    ##############################################################################
    img_input = Input(shape=(hpx,wpx , nclasses))

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

    out = Conv2D(nclasses, (1, 1), padding='same')(conv5)
    ##############################################################################

    model = get_segmentation_model(img_input ,  out ) # this would build the segmentation model
    latest_weights = find_latest_checkpoint(ckpath)
    if latest_weights is not None:
        model.load_weights(latest_weights)

    return model

def get_mininet_model(img, nclasses=3, ckpath = "ML/checkpoints_mosaic/mynet_arcjetCV"):
    """Get custom neural net model for model/shock/background segmentation

    Args:
        img (2D np array): opencv image
        nclasses (int, optional): Number of classes. Defaults to 3.
        ckpath (str, optional): Checkpoints path for saving model weights. Defaults to "ML/checkpoints_mosaic/mynet_arcjetCV".

    Returns:
        model: keras model object
    """
    cv.imwrite("frame.png", img)

    height = img.shape[0]
    width= img.shape[1]
    hpx = height- height%4
    wpx = width- width%4

    ##############################################################################
    img_input = Input(shape=(hpx,wpx , nclasses))

    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.5)(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.5)(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.5)(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)

    out = Conv2D(nclasses, (1, 1), padding='same')(conv5)
    ##############################################################################

    model = get_segmentation_model(img_input ,  out ) # this would build the segmentation model
    latest_weights = find_latest_checkpoint(ckpath)
    if latest_weights is not None:
        model.load_weights(latest_weights)

    return model

def cnn_apply(img, model):
    height = img.shape[0]
    width= img.shape[1]
    hpx = height- height%4
    wpx = width- width%4
    inimg = img[0:hpx,0:wpx,:]

    out = model.predict_segmentation(inp = inimg)
    outimg = np.zeros((height,width))
    outimg[0:hpx,0:wpx] = out

    return outimg

def train_model(model, frame_folder, mask_folder, epochs= 5, ckpath=None, LOAD=False):

    if LOAD:
        latest_weights = find_latest_checkpoint(ckpath)
        model.load_weights(latest_weights)

    model.train( 
        train_images =  frame_folder,
        train_annotations = mask_folder,
        checkpoints_path = ckpath , epochs=epochs,
        batch_size=2,
        do_augment=True
    )
    print('FINISHED TRAINING')

############################################################################
if __name__ == "__main__":
    from keras_segmentation.models.segnet import segnet
    from keras_segmentation.models.unet import unet
    import matplotlib.pyplot as plt
    from models import FrameMeta
    from utils.Functions import convert_mask_gray_to_BGR, cropBGR, cropGRAY,splitfn
    import os
    import pickle

    arcjetCVFolder = os.getcwd()+"/"
    orig_folder = arcjetCVFolder+"data/sample_frames/"
    mosaic_frames = arcjetCVFolder+"data/mosaic_frames/"
    mask_folder = arcjetCVFolder+"data/sample_masks/"
    mosaic_masks = arcjetCVFolder+"data/mosaic_masks/"
    video_folder= arcjetCVFolder+"data/video/"
    ml_model_name = "mynetRGB_256"
    checkpoint_folder = arcjetCVFolder+ "ML/checkpoints/" + ml_model_name
    #checkpoint_folder = "ML/checkpoints_mosaic/"

    TRAIN = False
    CHECK_CNN_MASKS = True    

    if TRAIN:
        train_frames = os.path.join(mosaic_frames,"train")
        train_masks = os.path.join(mosaic_masks,"train")
        val_frames = os.path.join(mosaic_frames,"validate")
        val_masks = os.path.join(mosaic_masks,"validate")

        files = glob(os.path.join(train_frames, "*.png"))
        img = cv.imread(files[0],1)
        # model = get_mininet_model(img,ckpath=checkpoint_folder)
        # model = unet(n_classes=3, input_height= 128, input_width=128)
        # latest_weights = find_latest_checkpoint(checkpoint_folder)
        # if latest_weights is not None:
        #     model.load_weights(latest_weights)
        model = get_unet_model(img,ckpath=checkpoint_folder)
        
        model.train(train_images =  train_frames,
                    train_annotations = train_masks,
                    validate = True,
                    batch_size=2,
                    val_batch_size=2,
                    val_images = val_frames, 
                    val_annotations = val_masks, 
                    checkpoints_path = checkpoint_folder, 
                    epochs=50,
                    do_augment=False, augmentation_name="aug_geometric")

        with open(ml_model_name+".pickle",'wb') as myhistory:
            pickle.dump(model.history.history.copy(), myhistory)

        for key in model.history.history.keys():
            plt.plot(model.history.history[key], label=key)
        plt.legend()
        plt.show()

    else:
        files = glob(orig_folder + "*.png")
        img = cv.imread(files[0],1)

    if CHECK_CNN_MASKS:
        for framepath in files:
            folder, name, ext = splitfn(framepath)
            print(name)
            maskpath = os.path.join(mask_folder, name+ext)
            metapath = os.path.join(folder,name+".meta")

            frame = cv.imread(framepath,1)
            #frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            mask = cv.imread(maskpath,0)
            maskBGR = convert_mask_gray_to_BGR(mask)
            meta = FrameMeta(metapath)

            crop = meta.crop_range()
            img_crop = cropBGR(frame, crop)
            mask_crop= cropBGR(maskBGR, crop)

            #model = get_mininet_model(img_crop,ckpath=checkpoint_folder)
            model = get_unet_model(img_crop,ckpath=checkpoint_folder)
    
            plot_model(model, to_file=arcjetCVFolder+ml_model_name+".png", show_shapes=True)

            ML_mask = cnn_apply(img_crop,model)
            ML_mask = convert_mask_gray_to_BGR(ML_mask)

            alpha = .5
            beta = (1.0 - alpha)
            dst = cv.addWeighted(img_crop, alpha, ML_mask, beta, 0.0)
            plt.subplot(121)
            plt.title("ML mask")
            plt.imshow(ML_mask)

            dst = cv.addWeighted(img_crop, alpha, mask_crop, beta, 0.0)
            plt.subplot(122)
            plt.title("Training mask")
            plt.imshow(dst)

            plt.show()


    
    
