import cv2
import os
from tqdm import tqdm
import math
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from facenet_pytorch.models.mtcnn import MTCNN
mtcnn_detector = MTCNN(margin=0, thresholds=[0.65, 0.75, 0.75], device="cpu")

def remove_eyes(image, landmarks, mask=None, cutout_fill=0, threshold=0.3):
    if landmarks is None:
        return image

    (x1, y1), (x2, y2) = landmarks[:2]
    line = cv2.line(
        np.zeros_like(image[..., 0]), (x1, y1), (x2, y2), color=(1), thickness=2
    )
    w = _distance((x1, y1), (x2, y2))

    return _remove(image, mask, w, line, cutout_fill, threshold)


# generate a mask, instead of manipulating the original image
def _remove(image, mask, w, line, cutout_fill, threshold):

    image = np.ones_like(image)
    if mask is not None:
        mask_ones = np.count_nonzero(mask == 1)
    for i in range(3, 7):  # Try multiple times to get max overlap below threshold
        line_ = binary_dilation(line, iterations=int(w // i))
        if mask is not None:
            cut = np.bitwise_and(line_, mask)
            cut_ones = np.count_nonzero(cut == 1)
            if (cut_ones / mask_ones) > threshold:
                continue
        if cutout_fill == -1:
            image[line_, :] = np.random.randint(0, 255, image[line_, :].shape)
        else:
            image[line_, :] = cutout_fill
        break
    return image


def _distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def cut_eyes_image(img_ori):
    # img_ori should be nparray and has RGB channels
    # return a mask with 3 channels

    img = Image.fromarray(img_ori)
    _, _, landmarks = mtcnn_detector.detect(img, landmarks=True)
    if landmarks is not None:
        landmarks = np.around(landmarks[0]).astype(np.int16)
        
        aug = img_ori.copy()
        eyes_mask = remove_eyes(aug, landmarks=landmarks, mask=None)
    else:
        eyes_mask = np.ones_like(img_ori)

    return eyes_mask


# ================================================================================
if __name__=='__main__':
    folder = r'/sdata/xianyun.sun/casia_MFSD_img/test_release_crop'
    subfolder = os.listdir(folder)

    for f in tqdm(subfolder):
        #print('in:', f)
        full_folder = os.path.join(folder, f)
        img_list = os.listdir(full_folder)

        for i in img_list:
            if i.split('.')[-1]=='png':
                img = cv2.imread(os.path.join(full_folder, i))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask = cut_eyes_image(img)

                np.save(os.path.join(full_folder, i.replace('.png', '_mask.npy')), mask)
            

    
