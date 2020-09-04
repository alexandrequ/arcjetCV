from PIL import Image
import os
import cv2

path = "train_frames_original/"
idx = 0
for element in os.listdir(path):
    if element.endswith('.png'):
        #os.mkdir("dataset/train_masks_mosaic/" + str(element))
        img = Image.open(path + element)
        width, height = img.size
        w = 128
        while (w < width):
            h = 128
            while (h < height):
                name  = element[:-4]
                img.crop((w-128, h-128, w, h)).save("dataset/train_frames_mosaic/" + str(name) +"_"+ str(w) +"_"+ str(h) + ".png")
                h = h + 128
            w = w + 128
    idx = idx + 1
