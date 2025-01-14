import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import cv2

from utils import noise_reduction
from torchvision import transforms
import random
from datetime import datetime

logging.getLogger('PIL').setLevel(logging.WARNING)

def pepper_noisy(img):
    is_pil = False
    if isinstance(img, Image.Image):
        is_pil = True
        img = np.array(img)
    
    assert 1 < len(img.shape) < 4, "image has 1 or more than 3 channels"
    hsv = False
    if len(img.shape) == 2:
        row, col = img.shape
    elif len(img.shape) == 3:
        hsv = True
        row, col, _ = img.shape
        tmp = img.copy()
        img = img[:, :, 2]
        
    
    amount = (random.random() - 0.25) / 3
    if amount > 0.15:
        amount = 0.15
    elif amount < 0:
        amount = 0
    num_pepper = int(np.ceil(amount * img.size))

    amount = (random.random() - 0.25) / 3
    if amount > 0.08:
        amount = 0.08
    elif amount < 0:
        amount = 0
    num_salt = int(np.ceil(amount * img.size))
        
    out = np.copy(img)

    map_pepper_salt = [2] * num_salt + [1] * num_pepper + [0] * (row * col - num_pepper - num_salt)
    random.shuffle(map_pepper_salt)
    map_pepper_salt = np.array(map_pepper_salt).reshape((row, col))
    out[map_pepper_salt == 2] = (out[map_pepper_salt == 2] * 1.6).astype(np.uint8)
    out[map_pepper_salt == 1] = (out[map_pepper_salt == 1] * 0.5).astype(np.uint8)

    if hsv:
        tmp[:, :, 2] = out
        out = tmp
    
    if is_pil:
        if hsv:
            return Image.fromarray(out, mode = "HSV")
        else:
            return Image.fromarray(out)
    else:
        return out

def gamma_correction(img):
    is_pil = False
    if isinstance(img, Image.Image):
        is_pil = True
        img = np.array(img)

    hsv = False
    if len(img.shape) == 3:
        hsv = True
        tmp = img.copy()
        img = img[:, :, 2]

    img = img.astype(float)
    m = img.min()
    img = img - m
    r = img.max()

    gamma = random.random() * 2.5
    gamma = gamma if gamma > 0.5 else 0.5
    table = np.array([((i / float(r)) ** gamma) * r for i in np.arange(0, r + 1)])

    result = table[img.astype(np.uint8)]
    result = result + m
    result = np.clip(result, 0, 255).astype(np.uint8)

    if hsv:
        tmp[:, :, 2] = result
        result = tmp
    
    if is_pil:
        if hsv:
            return Image.fromarray(result, mode = "HSV")
        else:
            return Image.fromarray(result)
    else:
        return result

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    """
    images_dir: 訓練用的影像路徑
    mask_dir: 訓練用的 mask ，其中 0 代表背景， 1 代表 GA ， 2 跟 3 都代表聲帶
    scale: 影像讀取後縮放的尺寸，基本不會用到
    mask_suffix: mask 的檔名後面的接尾，我覺得沒必要用到
    augment: 是否在訓練時進行 data augmentation
    out_img: 我們希望 dataset 輸出甚麼格式的影像，也就是要為給模型甚麼格式的影像，可以是 HSV 或 gray
    """
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', augment: bool = True, out_img: str = "HSV"):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.augment = augment
        self.out_img = out_img

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        # logging.info(f'Creating dataset with {len(self.ids)} examples')
        # logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        # logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def randTransform(self, img, mask):
        rand_seed = datetime.now().timestamp()
        rand_seed += random.randint(0, 2 ** 16 - 1)
        random.seed(rand_seed)
        transform_seed = random.randint(0, 2 ** 16 - 1)
        # logging.debug(f"randTransform, rand_seed = {rand_seed}, transform_seed = {transform_seed}")

        transform_img = transforms.Compose([
            transforms.RandomAffine(degrees = (-30, 30), shear = (-30, 30), interpolation = Image.BICUBIC),
            transforms.RandomHorizontalFlip(p = 0.5),
            # transforms.GaussianBlur(kernel_size = 9, sigma = (0.1, 5.0)),
            ## 0113新增
            # transforms.RandomVerticalFlip(p=0.3),  # 新增垂直翻轉
            # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 調整亮度、對比度、飽和度、色相
            # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),  # 新增透視變換
            # transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),  # 隨機擦除
            ##
            pepper_noisy,
            gamma_correction,
            # transforms.GaussianBlur(kernel_size = 9, sigma = (1.0, 1.0)),
        ])
        transform_mask = transforms.Compose([
            transforms.RandomAffine(degrees = (-30, 30), shear = (-30, 30), interpolation = Image.NEAREST),
            transforms.RandomHorizontalFlip(p = 0.5),
            ##0113新增
            # transforms.RandomVerticalFlip(p=0.3),  # 確保 mask 與影像一致
            # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        ])

        random.seed(rand_seed)
        torch.random.manual_seed(transform_seed)
        img = transform_img(img)

        random.seed(rand_seed)
        torch.random.manual_seed(transform_seed)
        mask = transform_mask(mask)

        return img, mask

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
            
        try:
            assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
            assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
            mask = load_image(mask_file[0])
            mask = np.array(mask)

            # 3 跟 2 都是聲帶，網路上的資料集有區分左聲帶以及右聲帶
            # 但我們不需要區分，因此一律設定成 2
            mask[mask == 3] = 2
            mask = Image.fromarray(mask)

            # 讀檔，其中 self.input_format 代表我們的模型實際上要吃的圖片 format 
            # (不是現在讀進來的 img 的 format ，讀進來的彩圖一律是 RGB )
            img = load_image(img_file[0])
            if self.out_img == "HSV":
                img = img.convert("HSV")
            elif self.out_img == "gray":
                img = img.convert('L')
            else:
                print("Unknow self.input_format. Should be either 'HSV' or 'gray'.")
                exit(1)
            
            # import matplotlib.pyplot as plt
            # plt.imshow(img)
            # plt.show()

            # data augmentation
            if self.augment:
                img, mask = self.randTransform(img, mask)

            # 透過 mask 找出 bounding box 
            # 將 x 軸以及 y 軸分別加總，如果大於 0 則代表這個方位有 GA
            # 透過 np where 以及 np diff 找出從沒有 GA 變化成有 GA 的交界處
            # 這個位置就是 bounding box
            tmp = np.asarray(mask)
            x = np.sum(tmp, axis = 0)
            y = np.sum(tmp, axis = 1)
            xpoints = np.where(np.diff(x > 0) != 0)[0]
            ypoints = np.where(np.diff(y > 0) != 0)[0]
                
            # 其中如果 xpoints, ypoints 的長度 2 的話，代表 ROI 有其中一邊在影像的邊緣處
            # 我們將他補成兩個，以利後續的 cropping 
            if len(xpoints) < 2:
                if x[-1] > 0:
                    xpoints = np.append(xpoints, len(x))
                else:
                    xpoints = np.append(0, xpoints)
            if len(ypoints) < 2:
                if y[-1] > 0:
                    ypoints = np.append(ypoints, len(y))
                else:
                    ypoints = np.append(0, ypoints)
                
            if len(xpoints) > 2:
                xpoints = [min(xpoints), max(xpoints)]
            if len(ypoints) > 2:
                ypoints = [min(ypoints), max(ypoints)]
                    
            w = xpoints[1] - xpoints[0]
            h = ypoints[1] - ypoints[0]
            target_len = max(w, h)
            x_center = (xpoints[1] + xpoints[0]) / 2
            y_center = (ypoints[1] + ypoints[0]) / 2
            if self.augment:
                target_len *= random.uniform(0.4, 0.6)
            else:
                target_len *= 0.5

            xpoints[0] = x_center - target_len
            xpoints[1] = x_center + target_len
            ypoints[0] = y_center - target_len
            ypoints[1] = y_center + target_len

            xpoints = (int(xpoints[0]), int(xpoints[1]))
            ypoints = (int(ypoints[0]), int(ypoints[1]))
                
            assert ypoints[1] - ypoints[0] > 3 and xpoints[1] - xpoints[0] > 3, f"{img_file}, is too small."
            # logging.debug(f"{img_file}, is too small.")
                        
            # Crop handel with the problem of cropping an area outside an image automatically. It only remain the valid part.
            img = img.crop((xpoints[0], ypoints[0], xpoints[1], ypoints[1]))
            mask = mask.crop((xpoints[0], ypoints[0], xpoints[1], ypoints[1]))

            assert img.size == mask.size, \
                f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

            newImgSize = (128, 128)
            w, h = img.size
            newScale = min(newImgSize[0] / w, newImgSize[1] / h)
            img = img.resize((int(w * newScale), int(h * newScale)), resample = Image.BICUBIC)
            mask = mask.resize((int(w * newScale), int(h * newScale)), resample = Image.NEAREST)
            # fill the new image with color 0 (black)
            w, h = img.size
            newImg = Image.new(img.mode, newImgSize, 0)
            newImg.paste(img, (int((128 - w) / 2), int((128 - h) / 2)))
            newMask = Image.new(mask.mode, newImgSize, 0)
            newMask.paste(mask, (int((128 - w) / 2), int((128 - h) / 2)))
            newMask = np.array(newMask)
            newMask[newMask == 2] = 0
            newMask = Image.fromarray(newMask)
            

            if self.augment:
                scale = random.random() * 10 / 2
                scale = scale if scale > 1 else 1
                newImg = newImg.resize((int(newImgSize[0] / scale), int(newImgSize[1] / scale)))
                newImg = newImg.resize(newImgSize)
            
            newImg = np.array(newImg)


            # plt.subplot(211)
            # plt.imshow(Image.fromarray(newImg, mode = "HSV"))
            if self.out_img == "HSV":
                tmp = newImg.copy()
                newImg = newImg[:, :, -1]
            elif self.out_img == "gray":
                pass
            else:
                print("Unknow self.input_format. Should be either 'HSV' or 'gray'.")
                exit(1)

            # 進行半自動化的 gamma correction
            img_median = np.median(newImg[newImg > 0])
            if img_median < 60 and img_median > 40:
                # print("median : 40 ~ 60")
                newImg = noise_reduction.gamma_correction(newImg, gamma = 0.5)
                img_median = np.median(newImg[newImg > 0])
            elif img_median < 40 and img_median > 20:
                # print("median : 20 ~ 40")
                newImg = noise_reduction.gamma_correction(newImg, gamma = 0.4)
                img_median = np.median(newImg[newImg > 0])
            if img_median < 20:
                # print("median : 0 ~ 20")
                newImg = noise_reduction.gamma_correction(newImg, gamma = 0.3)
                img_median = np.median(newImg[newImg > 0])
            
            # smoothing
            k_size = 7
            newImg = cv2.bilateralFilter(newImg, d = k_size, sigmaColor = 100, sigmaSpace = 100)

            # normalization
            newImg = newImg.astype(float)
            img_min = np.percentile(newImg[newImg > img_median * 0.2], 2)
            newImg = newImg - img_min
            img_max = np.percentile(newImg, 98)
            newImg[newImg > img_max] = img_max
            newImg = ((newImg / img_max) * 255)
            newImg[newImg < 0] = 0
            newImg = newImg.astype(np.uint8)
            
            if self.out_img == "HSV":
                tmp[:, :, -1] = newImg
                newImg = tmp
            elif self.out_img == "gray":
                pass
            else:
                print("out_img should be either 'HSV' or 'gray'")
                exit(1)
            # plt.subplot(212)
            # plt.imshow(Image.fromarray(newImg, mode = "HSV"))
            # plt.show()


            if self.out_img == "HSV":
                newImg = Image.fromarray(newImg, mode = "HSV")
            elif self.out_img == "gray":
                newImg = Image.fromarray(newImg)
            else:
                print("out_img should be either 'HSV' or 'gray'")
                exit(1)


            newImg = self.preprocess(self.mask_values, newImg, self.scale, is_mask=False)
            newMask = self.preprocess(self.mask_values, newMask, self.scale, is_mask=True)


            return {
                'image': torch.as_tensor(newImg.copy()).float().contiguous(),
                'mask': torch.as_tensor(newMask.copy()).long().contiguous(),
                'filename': f"{name}.png"  # 添加文件名
            }
        except:
            print(f"error occur in : {img_file}")
            return self.__getitem__(random.randrange(0, self.__len__()))


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
