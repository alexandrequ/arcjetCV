# increase the training data

import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import numpy as np 
seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])
img = cv2.imread("img001_2.png")
seg = cv2.imread("img001.png")


aug_det = seq.to_deterministic()
image_aug = aug_det.augment_image( img )

segmap = ia.SegmentationMapOnImage( seg , nb_classes=np.max(seg)+1 , shape=img.shape )
segmap_aug = aug_det.augment_segmentation_maps( segmap )
segmap_aug = segmap_aug.get_arr_int()
cv2.imwrite('img902_1.png', image_aug)
cv2.imwrite('img902.png', segmap_aug)
