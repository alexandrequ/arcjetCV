import cv2
import os

#print(os.listdir('test/'))
for file in os.listdir('train_masks_aug'):
    if not file.startswith('.'):
        print(file)
        img=cv2.imread('train_masks_aug/'+file)
        height, width, channels = img.shape

        white = [255,255,255]
        black = [0,0,0]

        for x in range(0,width):
            for y in range(0,height):
                channels_xy = img[y,x]
                if all(channels_xy == white):
                    img[y,x] = 2

                elif all(channels_xy == black):
                    img[y,x] = 0
                else:
                    img[y,x] = 1


        cv2.imwrite('train_masks_aug/'+file,img)
