import cv2
import numpy as np
from PIL import Image

# def gamma(image, gamma=0.7):
#     orig_shape = image.shape
#     m = image.min()
#     image = image - m
#     r = image.max() - image.min()
#     table = np.array([((i / float(r)) ** gamma) * r for i in np.arange(0, r + 1)])
#     result = table[image]
#     result = (result + m).astype(int)
#     return result

# def localAutoGamma(img):
#     is_pil = False
#     if isinstance(img, Image.Image):
#         is_pil = True
#         img = np.array(img)

#     w, h = img.shape
#     # should implemented
#     # segment the image into diffrent region and perform auto gamma on them seperately. 
#     # the target mean should be determined based on the original image's mean.

#     if is_pil:
#         return Image.fromarray(img)
#     else:
#         return img

# def AutoGamma(img, targetMean = 64):
#     import math
#     currMean = img.mean()
#     gammaValue = math.log10(targetMean) / math.log10(currMean)
#     img = gamma(img, gamma = gammaValue)

#     return img


def opening(img, k_size = 3):
    is_pil = False
    if isinstance(img, Image.Image):
        is_pil = True
        img = np.array(img)
    kernel = np.ones((k_size, k_size),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    if is_pil:
        return Image.fromarray(img)
    else:
        return img

def closing(img, k_size = 3):
    is_pil = False
    if isinstance(img, Image.Image):
        is_pil = True
        img = np.array(img)
    kernel = np.ones((k_size, k_size),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    if is_pil:
        return Image.fromarray(img)
    else:
        return img

def erosion(img, k_size = 3, iterations = 1):
    is_pil = False
    if isinstance(img, Image.Image):
        is_pil = True
        img = np.array(img)
    kernel = np.ones((k_size, k_size),np.uint8)
    img = cv2.erode(img, kernel, iterations = iterations)
    
    if is_pil:
        return Image.fromarray(img)
    else:
        return img

def dilation(img, k_size = 3, iterations = 1):
    is_pil = False
    if isinstance(img, Image.Image):
        is_pil = True
        img = np.array(img)
    kernel = np.ones((k_size, k_size),np.uint8)
    img = cv2.dilate(img, kernel, iterations = iterations)
    
    if is_pil:
        return Image.fromarray(img)
    else:
        return img

def _lightenGray_get_min_max(gray):
    is_pil = False
    if isinstance(gray, Image.Image):
        is_pil = True
        gray = np.array(gray)
        
    assert len(gray.shape) == 2, "Image is not in gray scale"

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.reshape((-1))
    grayMin, grayMax = 1, 254
    for i in range(255):
        if hist[i:].sum() / hist.sum() > 0.99:
            grayMin = i
        if hist[:i].sum() / hist.sum() > 0.95:
            grayMax = i
            break
    return [grayMin, grayMax]

def lightenGray(gray, min_max = None):
    is_pil = False
    if isinstance(gray, Image.Image):
        is_pil = True
        gray = np.array(gray)
        
    assert len(gray.shape) == 2, "Image is not in gray scale"
    
    if not min_max:
        min_max = _lightenGray_get_min_max(gray)

    assert len(min_max) == 2

    grayMin = min_max[0]
    grayMax = min_max[1]

    gray = gray.astype(np.int16)
    gray[gray < grayMin] = grayMin
    gray -= grayMin
    grayMax -= grayMin
    gray[gray > grayMax] = grayMax
    gray = (gray / grayMax * 255).astype(np.uint8)
        
    if is_pil:
        return Image.fromarray(gray)
    else:
        return gray

def imageRescale(img):
    is_np = False
    if isinstance(img, np.ndarray):
        is_np = True
        img = Image.fromarray(img)
        
    newImgSize = (128, 128)
    
    old_w, old_h = img.size
    newScale = min(newImgSize[0] / old_w, newImgSize[1] / old_h)
    img = img.resize((int(old_w * newScale), int(old_h * newScale)), resample = Image.BICUBIC)
    
    w, h = img.size
    newImg = Image.new(img.mode, newImgSize, 255)
    newImg.paste(img, (int((128 - w) / 2), int((128 - h) / 2)))
    newImg = np.array(newImg)
    
    if is_np:
        return np.array(newImg)
    else:
        return newImg

def findRed(img):
    img = img.astype(np.int16)
    
    redMap = (img[:, :, 0] - (img[:, :, 1] + img[:, :, 2]) / 2).copy()
    redMap[redMap < 0] = 0
    redMap = redMap.astype(np.uint8)
    img = img.astype(np.uint8)
    
    redThreshold = np.percentile(redMap, 80)
    
    redMap[redMap >= redThreshold] = 255
    redMap[redMap < redThreshold] = 0
    
    return redMap

def img_set_average(img, average):
    is_pil = False
    if isinstance(img, Image.Image):
        is_pil = True
        img = np.array(img)
    
    img = img.astype(np.int32)
    img -= int(img.mean() - average)
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    
    if is_pil:
        return Image.fromarray(img)
    else:
        return img