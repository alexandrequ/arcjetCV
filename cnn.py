# import opencv
import cv2 as cv 
import numpy as np

# import ML modules
from keras.models import Input
from keras.layers import Dropout,concatenate,UpSampling2D
from keras.layers import Conv2D, MaxPooling2D
from keras_segmentation.train import find_latest_checkpoint
from keras_segmentation.models.model_utils import get_segmentation_model

def get_unet_model(img, nclasses=3, ckpath = "ML/checkpoints_mosaic/mynet_arcjetCV"):

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

    out = Conv2D( nclasses, (1, 1) , padding='same')(conv5)
    ##############################################################################

    model = get_segmentation_model(img_input ,  out ) # this would build the segmentation model
    latest_weights = find_latest_checkpoint(ckpath)
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


def get_mosaic_set(inpath, outpath, regex = "*.png"):
    from glob import glob
    idx = 0
    paths = glob(inpath+regex)
    for element in paths:
        #os.mkdir("dataset/train_masks_mosaic/" + str(element))
        img = cv.imread(inpath + element)
        width, height = img.size
        w = 128
        while (w < width):
            h = 128
            while (h < height):
                name  = element[:-4]
                img.crop((w-128, h-128, w, h)).save(outpath + str(name) +"_"+ str(w) +"_"+ str(h) + ".png")
                h = h + 128
            w = w + 128
        idx = idx + 1

############################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from models import FrameMeta
    from utils.Functions import convert_mask_gray_to_BGR, cropBGR, cropGRAY

    orig_folder = "/home/magnus/Desktop/NASA/arcjetCV/data/sample_frames/"
    mask_folder = "/home/magnus/Desktop/NASA/arcjetCV/data/sample_masks/"
    video_folder= "/home/magnus/Desktop/NASA/arcjetCV/data/video/"

    vname = "IHF338Run003_WestView_3"
    vname = "AHF335Run001_EastView_5"
    vname = "IHF338Run002_WestView_1"
    ext = ".mp4"

    for n in range(0,80):
        framepath = orig_folder+"frame_%04d.png"%(n)
        maskpath = mask_folder+ "frame_%04d.png"%(n)
        metapath = orig_folder+ "frame_%04d.meta"%(n)

        # mask = folder+ name + ext
        # paths = glob(mask)
        frame = cv.imread(framepath,1)
        mask = cv.imread(maskpath,0)
        maskBGR = convert_mask_gray_to_BGR(mask)
        meta = FrameMeta(metapath)

        crop = meta.crop_range()
        img_crop = cropBGR(frame, crop)
        mask_crop= cropBGR(maskBGR, crop)

        UNet = get_unet_model(img_crop)

        ML_mask = cnn_apply(img_crop,UNet)
        ML_mask = convert_mask_gray_to_BGR(ML_mask)

        alpha = .5
        beta = (1.0 - alpha)
        dst = cv.addWeighted(img_crop, alpha, ML_mask, beta, 0.0)
        plt.subplot(121)
        plt.title("ML mask")
        plt.imshow(dst)

        dst = cv.addWeighted(img_crop, alpha, mask_crop, beta, 0.0)
        plt.subplot(122)
        plt.title("Training mask")
        plt.imshow(dst)

        plt.show()


    
    