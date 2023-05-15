import cv2
import os
import numpy as np
from tqdm import tqdm


folder = r'./PAs/Webcam_ReplayAttack' # change this folder to include all images in the training set
img_list = os.listdir(folder)
for i in tqdm(img_list):
    img_name = i.split('.')[0]
    img = cv2.imread(os.path.join(folder, i), cv2.IMREAD_COLOR)
    cv2.imwrite(os.path.join(folder, img_name+'_50.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    cv2.imwrite(os.path.join(folder, img_name+'_25.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 25])
