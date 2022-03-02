#### mf-ssl preprocessing stuff - https://github.com/numpy/numpy-tutorials/blob/main/content/tutorial-x-ray-image-processing.md
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

def create_mfs(img, aug):
    if aug == 'fourier':
        fourier_gaussian = ndimage.fourier_gaussian(xray_image, sigma=0.05)

        x_prewitt = ndimage.prewitt(fourier_gaussian, axis=0)
        y_prewitt = ndimage.prewitt(fourier_gaussian, axis=1)

        xray_image_canny = np.hypot(x_prewitt, y_prewitt)

        xray_image_canny *= 255.0 / np.max(xray_image_canny)
        return xray_image_canny

    elif aug == "gaussian":
        x_ray_image_gaussian_gradient = ndimage.gaussian_gradient_magnitude(xray_image, sigma=2)
        return x_ray_image_gaussian_gradient

    elif aug == "sobel":
        x_sobel = ndimage.sobel(xray_image, axis=0)
        y_sobel = ndimage.sobel(xray_image, axis=1)

        xray_image_sobel = np.hypot(x_sobel, y_sobel)

        xray_image_sobel *= 255.0 / np.max(xray_image_sobel)
        xray_image_sobel = xray_image_sobel.astype("float32")
        return xray_image_sobel
    
    elif aug == "laplace":
        xray_image_laplace_gaussian = ndimage.gaussian_laplace(xray_image, sigma=1)
        return xray_image_laplace_gaussian



if __name__ == "__main__":

    df = pd.read_csv("/MULTIX/DATA/nccid_preprocessed.csv")
    img_path = df['dgx_structured_path'][0]
    xray_image = cv2.imread(img_path)

    xray_image_laplace_gaussian = ndimage.gaussian_laplace(xray_image, sigma=1)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

    axes[0].set_title("Original")
    axes[0].imshow(xray_image, cmap="gray")
    axes[1].set_title("Laplacian-Gaussian (edges)")
    axes[1].imshow(xray_image_laplace_gaussian, cmap="gray")
    for i in axes:
        i.axis("off")
    plt.show()

    x_ray_image_gaussian_gradient = ndimage.gaussian_gradient_magnitude(xray_image, sigma=2)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

    axes[0].set_title("Original")
    axes[0].imshow(xray_image, cmap="gray")
    axes[1].set_title("Gaussian gradient (edges)")
    axes[1].imshow(x_ray_image_gaussian_gradient, cmap="gray")
    for i in axes:
        i.axis("off")
    plt.show()

    x_sobel = ndimage.sobel(xray_image, axis=0)
    y_sobel = ndimage.sobel(xray_image, axis=1)

    xray_image_sobel = np.hypot(x_sobel, y_sobel)

    xray_image_sobel *= 255.0 / np.max(xray_image_sobel)

    print("The data type - before: ", xray_image_sobel.dtype)

    xray_image_sobel = xray_image_sobel.astype("float32")

    print("The data type - after: ", xray_image_sobel.dtype)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))

    axes[0].set_title("Original")
    axes[0].imshow(xray_image, cmap="gray")
    axes[1].set_title("Sobel (edges) - grayscale")
    axes[1].imshow(xray_image_sobel, cmap="gray")
    axes[2].set_title("Sobel (edges) - CMRmap")
    axes[2].imshow(xray_image_sobel, cmap="CMRmap")
    for i in axes:
        i.axis("off")
    plt.show()

    fourier_gaussian = ndimage.fourier_gaussian(xray_image, sigma=0.05)

    x_prewitt = ndimage.prewitt(fourier_gaussian, axis=0)
    y_prewitt = ndimage.prewitt(fourier_gaussian, axis=1)

    xray_image_canny = np.hypot(x_prewitt, y_prewitt)

    xray_image_canny *= 255.0 / np.max(xray_image_canny)

    print("The data type - ", xray_image_canny.dtype)

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 15))

    axes[0].set_title("Original")
    axes[0].imshow(xray_image, cmap="gray")
    axes[1].set_title("Canny (edges) - prism")
    axes[1].imshow(xray_image_canny, cmap="prism")
    axes[2].set_title("Canny (edges) - nipy_spectral")
    axes[2].imshow(xray_image_canny, cmap="nipy_spectral")
    axes[3].set_title("Canny (edges) - terrain")
    axes[3].imshow(xray_image_canny, cmap="terrain")
    for i in axes:
        i.axis("off")
    plt.show()