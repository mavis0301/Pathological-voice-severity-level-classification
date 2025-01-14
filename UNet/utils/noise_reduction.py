import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def denoise_enhance_edges(img, img_type = "gray"):
    is_pil = isinstance(img, Image.Image)
    if is_pil:
        img = np.array(img)

    if img_type != "gray":
        return img
    
    k_size = 7 if img.shape[0] * img.shape[1] < 400 else 11
    img = cv2.bilateralFilter(img, d = k_size, sigmaColor = 150, sigmaSpace = 150)
    if np.median(img) < 30:
        img = gamma_correction(img, gamma = 0.5)
    clahe = cv2.createCLAHE(clipLimit = 2, tileGridSize=(8, 8))
    img = clahe.apply(img)
    k_size = 5 if img.shape[0] * img.shape[1] < 400 else 9
    img = cv2.bilateralFilter(img, d = k_size, sigmaColor = 100, sigmaSpace = 100)

    if is_pil:
        return Image.fromarray(img)
    else:
        return img
    
def gamma_correction(img, gamma):
    is_pil = False
    if isinstance(img, Image.Image):
        is_pil = True
        img = np.array(img)

    img = img.astype(float)
    m = img[img > 0].min()
    img = img - m
    img[img < 0] = 0
    # r = min(img.max(), 80)
    r = img.max()
    img[img > r] = r

    table = np.array([((i / float(r)) ** gamma) * r for i in np.arange(0, r + 1)])

    result = table[img.astype(np.uint8)]
    result = result + m
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    if is_pil:
        return Image.fromarray(result)
    else:
        return result

"""
形態學上的 opening operation
"""
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

    gray = gray.astype(float)
    gray[gray < grayMin] = grayMin
    gray -= grayMin
    grayMax -= grayMin
    gray[gray > grayMax] = grayMax
    gray = (gray / grayMax * 255).astype(np.uint8)
        
    if is_pil:
        return Image.fromarray(gray)
    else:
        return gray