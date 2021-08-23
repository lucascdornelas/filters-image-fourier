import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist
from utils import *
import filters


if __name__ == '__main__':

    img_path = "new_data/monument.jpeg"
    
    img = imread(img_path)

    img_grey = rgb2gray(img)
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(img_grey, cmap='gray')

    img_grey_fourier = np.fft.fftshift(np.fft.fft2(img_grey))
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(np.log(abs(img_grey_fourier)), cmap='gray');

    fourier_masker(img)

    fourier_masker_hor(img)


    # fft_image_filtered = fft_img * filters.gaussianLP(5,fft_img.shape)
    # fft_image_filtered = fft_img * filters.gaussianHP(50,fft_img.shape) 
    # fft_image_filtered = fft_img * filters.idealFilterLP(50,fft_img.shape)
    # fft_image_filtered = fft_img * filters.idealFilterHP(50,fft_img.shape) 
    # fft_image_filtered = fft_img * filters.butterworthLP(5,fft_img.shape)
    # fft_image_filtered = fft_img * filters.butterworthHP(50,fft_img.shape) 

    fft_img = np.fft.ifftshift(fft_image_filtered)   

    image_fft_inverse(fft_img, img_path)

