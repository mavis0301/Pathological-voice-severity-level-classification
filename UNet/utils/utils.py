import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()

def polygons_to_mask(polygons, shape):
    
    mask = np.zeros(shape)
    cv2.fillPoly(mask, pts=polygons, color=1)
    
    return mask

def mask_to_polygons(mask):
#     mask = (mask == 1)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    normalized_polygons = []
    
    for contour in contours:
        polygon = contour.reshape(-1, 2)
        polygons.append(polygon)

        normalized_polygon = [[round(coord[0] / mask.shape[1] , 4), round(coord[1] / mask.shape[0] , 4)] for coord in polygon]
        normalized_polygons.append(normalized_polygon)
    
    return polygons, normalized_polygons