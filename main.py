import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def read_image(image_path):
    return plt.imread(image_path)

def display_images(images, titles):
    num_images = len(images)
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

def apply_gaussian_blur(grayscale_image):
    imgarray = np.array(grayscale_image)
    kernel = np.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]])
    img_height, img_width = imgarray.shape
    kernel_height, kernel_width = kernel.shape
        
    padded_img = np.pad(imgarray, ((1, 1), (1, 1)), mode='constant')
        
    blurred_img = np.zeros_like(imgarray)
        
    for i in range(img_height):
        for j in range(img_width):
            blurred_img[i, j] = np.sum(padded_img[i:i+kernel_height, j:j+kernel_width] * kernel)
    return blurred_img

def sobel_operators(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gradient_x = np.zeros_like(image, dtype=np.float32)
    gradient_y = np.zeros_like(image, dtype=np.float32)
    image_height, image_width = image.shape
    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            gradient_x[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * sobel_x)
            gradient_y[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * sobel_y)
    return gradient_x, gradient_y

def gradient_magnitude_direction(gradient_x, gradient_y):
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    direction = np.arctan2(gradient_y, gradient_x)
    return magnitude, direction

def non_max_suppression(magnitude, direction):
    suppressed_magnitude = np.zeros_like(magnitude)
    image_height, image_width = magnitude.shape
    direction_int= (np.round(direction * 4 / np.pi) % 4).astype(int)
    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            neighbor1, neighbor2 = 0, 0
            if direction_int[i, j] == 0:
                neighbor1 = magnitude[i, j-1]
                neighbor2 = magnitude[i, j+1]
            elif direction_int[i, j] == 1:
                neighbor1 = magnitude[i-1, j+1]
                neighbor2 = magnitude[i+1, j-1]
            elif direction_int[i, j] == 2:
                neighbor1 = magnitude[i-1, j]
                neighbor2 = magnitude[i+1, j]
            elif direction_int[i, j] == 3:
                neighbor1 = magnitude[i-1, j-1]
                neighbor2 = magnitude[i+1, j+1]
            if magnitude[i, j] >= neighbor1 and magnitude[i, j] >= neighbor2:
                suppressed_magnitude[i, j] = magnitude[i, j]
    return suppressed_magnitude

def double_thresholding(suppressed_magnitude, low_threshold_ratio=0.12, high_threshold_ratio=0.16):
    high_threshold = suppressed_magnitude.max() * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio
    strong_edges = suppressed_magnitude > high_threshold
    weak_edges = (suppressed_magnitude >= low_threshold) & (suppressed_magnitude <= high_threshold)
    return strong_edges, weak_edges

def hysteresis(strong_edges, weak_edges):
    image_height, image_width = strong_edges.shape
    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            if weak_edges[i, j]:
                if strong_edges[i-1:i+2, j-1:j+2].any():
                    strong_edges[i, j] = 1
                else:
                    weak_edges[i, j] = 0
    return strong_edges

def canny_edge_detection(image_path, kernel_size=5, sigma=1, low_threshold_ratio=0.05, high_threshold_ratio=0.09):
    # Task 1
    original_image = read_image(image_path)
    grayscale_image = np.mean(original_image, axis=2)

    # Task 2
    blurred_image = apply_gaussian_blur(grayscale_image)

    # Task 3
    gradient_x, gradient_y = sobel_operators(blurred_image)
    gradient_magnitude, gradient_direction = gradient_magnitude_direction(gradient_x, gradient_y)

    # Task 4
    suppressed_magnitude = non_max_suppression(gradient_magnitude, gradient_direction)

    # Task 5
    strong_edges, weak_edges = double_thresholding(suppressed_magnitude)

    # Task 6
    final_edges = hysteresis(strong_edges, weak_edges)
    
    display_images([original_image, final_edges], "Original vs. Final")

if __name__ == "__main__":
    image_path = 'chic.jpg'
    canny_edge_detection(image_path)