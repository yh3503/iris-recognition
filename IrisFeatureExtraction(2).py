import numpy as np
import cv2
from IrisNormalization import normalization

def defined_filter(x, y, f, delta_x, delta_y):
    g = (1 / (2 * np.pi * delta_x * delta_y)) * np.exp(-0.5 * ((x**2/delta_x**2) + (y**2/delta_y**2)))
    m1 = np.cos(2 * np.pi * f * np.sqrt(x**2 + y**2))
    return g * m1

def extract_features(iris_image, two_circle):
    normalized_img = normalization(iris_image, two_circle)
    # ROI
    height, width, _ = normalized_img.shape
    roi = normalized_img[:height//2, :]

    # two channels
    channels = [
        {'delta_x': 3, 'delta_y': 1.5},
        {'delta_x': 4.5, 'delta_y': 1.5}
    ]

    filtered_images = []

    # Apply filters on ROI
    for channel in channels:
        filter_response = np.zeros((height//2, width))
        for i in range(height//2):
            for j in range(width):
                x = j - width // 2
                y = i - height // 4  # Divide by 4 because the ROI is half the height
                filter_response[i, j] = defined_filter(x, y, 1.0/channel['delta_y'], channel['delta_x'], channel['delta_y'])

        # Convolve the ROI with filter response
        filtered_image = cv2.filter2D(roi, -1, filter_response)
        filtered_images.append(filtered_image)

    # Extract features from the filtered images
    features = []

    block_size = 8
    for filtered_image in filtered_images:
        for i in range(0, filtered_image.shape[0], block_size):
            for j in range(0, filtered_image.shape[1], block_size):
                block = filtered_image[i:i+block_size, j:j+block_size]
                m = np.mean(block)
                sigma = np.mean(np.abs(block - m))
                features.extend([m, sigma])

    feature_vector = np.array(features).T

    return feature_vector





###参考
def defined_filter(size, sigma_x, sigma_y):
    """
    Define a Gabor filter with the given size, sigma_x, and sigma_y.
    """
    f = 1.0 / sigma_y
    filter_mat = np.zeros((size, size))

    for xi in range(size):
        for yi in range(size):
            x = xi - size // 2
            y = yi - size // 2
            gaussian_value = 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-1.0 / 2 * (x**2 / sigma_x**2 + y**2 / sigma_y**2))
            M1 = np.cos(2 * np.pi * f * np.sqrt(x**2 + y**2))
            filter_mat[yi][xi] = gaussian_value * M1

    return filter_mat

def execute_feature_extraction(input):
    """
    Extract features from the input images using Gabor filters.
    """
    def feature_extraction(filtered_img_1, filtered_img_2):
        V = []
        for row_index in range(0, filtered_img_1.shape[0], 8):  # Use a stride of 8
            for col_index in range(0, filtered_img_1.shape[1], 8):  # Use a stride of 8
                # Process the first filtered image
                sub_region_vec1 = abs(filtered_img_1[row_index:row_index + 8, col_index:col_index + 8].ravel())
                # Calculate m and sigma as denoted in the paper
                m1 = sub_region_vec1.mean()
                sigma1 = 1 / 64 * (abs(sub_region_vec1 - m1).sum())
                V.append(m1)
                V.append(sigma1)

                # Process the second filtered image
                sub_region_vec2 = abs(filtered_img_2[row_index:row_index + 8, col_index:col_index + 8].ravel())
                # Calculate m and sigma as denoted in the paper
                m2 = sub_region_vec2.mean()
                sigma2 = 1 / 64 * (abs(sub_region_vec2 - m2).sum())
                V.append(m2)
                V.append(sigma2)
        return V

    output = []
    filter1 = defined_filter(size=3, sigma_x=3, sigma_y=1.5)  # 3x3 defined Gabor filter
    filter2 = defined_filter(size=3, sigma_x=4.5, sigma_y=1.5)  # 3x3 defined Gabor fi


