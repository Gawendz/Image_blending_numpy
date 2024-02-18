import numpy as np
import matplotlib.pyplot as plt

def convolve(image, kernel):
    return np.sum(np.multiply(image, kernel), axis=(2, 3))

def resize_image(image, target_height, target_width):
    i, j = np.meshgrid(np.linspace(0, target_height - 1, target_height),
                       np.linspace(0, target_width - 1, target_width), indexing='ij')

    i_orig = i * (image.shape[0] - 1) / (target_height - 1)
    j_orig = j * (image.shape[1] - 1) / (target_width - 1)

    i_low, j_low = i_orig.astype(int), j_orig.astype(int)
    i_high, j_high = np.minimum(i_low + 1, image.shape[0] - 1), np.minimum(j_low + 1, image.shape[1] - 1)

    weights = (i_high - i_orig) * (j_high - j_orig)

    scaled_image = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)

    for c in range(image.shape[2]):
        scaled_image[..., c] = (
            weights * image[i_low, j_low, c] +
            (i_high - i_orig) * (j_orig - j_low) * image[i_low, j_high, c] +
            (i_orig - i_low) * (j_high - j_orig) * image[i_high, j_low, c] +
            (i_orig - i_low) * (j_orig - j_low) * image[i_high, j_high, c]
        )

    return scaled_image.astype(np.uint8)

def gaussian_kernel(size, sigma=1.0):
    x, y = np.meshgrid(np.arange(-size//2 + 1, size//2 + 1), np.arange(-size//2 + 1, size//2 + 1))
    kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)

def gaussian_blur(image, sigma=1.0):
    kernel = gaussian_kernel(5, sigma)
    kernel = kernel.reshape(1, 1, kernel.shape[0], kernel.shape[1])

    if len(image.shape) == 3:
        blurred_image = np.zeros_like(image, dtype=np.float64)
        for channel in range(image.shape[2]):
            blurred_image[..., channel] = convolve(image[..., channel, np.newaxis, np.newaxis], kernel)
    else:
        blurred_image = convolve(image[..., np.newaxis, np.newaxis], kernel)

    return blurred_image.squeeze()

def image_blend(img1, img2):
    target_height, target_width = 1000, 1000
    img1_resized = resize_image(img1, target_height, target_width)
    img2_resized = resize_image(img2, target_height, target_width)

    pyramid_img1 = [img1_resized]
    pyramid_img2 = [img2_resized]

    for _ in range(5):
        img1_resized = gaussian_blur(img1_resized)
        img2_resized = gaussian_blur(img2_resized)
        img1_resized = resize_image(img1_resized, target_height, target_width)
        img2_resized = resize_image(img2_resized, target_height, target_width)
        pyramid_img1.append(img1_resized)
        pyramid_img2.append(img2_resized)

    blended_pyramid = []
    for level in range(len(pyramid_img1)):
        blended_image = np.zeros_like(pyramid_img1[level], dtype=np.float64)

        blended_image[:, :, :blended_image.shape[2] // 2] = 0.5 * pyramid_img1[level][:, :, :blended_image.shape[2] // 2] + \
                                                             0.5 * pyramid_img2[level][:, :, :blended_image.shape[2] // 2]
        blended_image[:, :, blended_image.shape[2] // 2:] = 0.5 * pyramid_img1[level][:, :, blended_image.shape[2] // 2:] + \
                                                            0.5 * pyramid_img2[level][:, :, blended_image.shape[2] // 2:]
        blended_pyramid.append(blended_image)

    # Reconstruct the blended image from the blended pyramid
    blended_image = blended_pyramid[-1]
    for level in range(len(blended_pyramid) - 2, -1, -1):
        blended_image = resize_image(blended_image, blended_pyramid[level].shape[0], blended_pyramid[level].shape[1])
        blended_image[:, :, :blended_image.shape[2] // 2] = 0.5 * blended_image[:, :, :blended_image.shape[2] // 2] + \
                                                             0.5 * blended_pyramid[level][:, :, :blended_pyramid[level].shape[2] // 2]
        blended_image[:, :, blended_image.shape[2] // 2:] = 0.5 * blended_image[:, :, blended_image.shape[2] // 2:] + \
                                                            0.5 * blended_pyramid[level][:, :, blended_pyramid[level].shape[2] // 2:]

    # Display the result
    plt.imshow(blended_image)
    plt.title('Image Blend')
    plt.show()

# Load images
img1 = plt.imread("PG.jpg")
img2 = plt.imread("Orange.jpg")

# Perform image blending
image_blend(img1, img2)
