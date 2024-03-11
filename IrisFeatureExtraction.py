import numpy as np
import cv2


# Define the G function
def G(x, y, sigma_x, sigma_y):
    return (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(-0.5 * ((x ** 2 / sigma_x ** 2) + (y ** 2 / sigma_y ** 2)))


# Define the M1 function (Defined filter)
def M1(x, y, f):
    return np.cos(2 * np.pi * f * np.sqrt(x ** 2 + y ** 2))


# Define the M2 function (Gabor filter)
def M2(x, y, f, theta):
    return np.cos(2 * np.pi * f * (x * np.cos(theta) + y * np.sin(theta)))


# Define a function to create the filter kernel
# 首先，它创建了一个大小为kernel_size的零矩阵kern，它将用来保存滤波核的值。
# 然后，它计算滤波器核的中心位置center。这是重要的，因为滤波器核通常围绕中心对称，以确保在滤波时不会对图像产生偏移。
# 接着，函数通过一个双层循环遍历kern的每个元素，对于每个位置(x, y)：
# 计算x_shifted 和 y_shifted，它们是相对于中心的坐标，因为滤波器是以中心对称的。
# 如果提供了theta（对于Gabor滤波器），函数会使用G（高斯函数）和M_func（调制函数，即M2）来计算当前位置的值。
# 如果没有提供theta（对于自定义滤波器），函数会假定调制函数只需要x_shifted、y_shifted和f这些参数（即M1）。
# 最后，kern中的每个元素都是通过高斯函数G和调制函数M_func的乘积得到的，这正是根据指令构建的滤波器模型。
def create_filter_kernel(M_func, f, sigma_x, sigma_y, theta=None, kernel_size=(31, 31)):
    kern = np.zeros(kernel_size)
    center = (kernel_size[0] // 2, kernel_size[1] // 2)
    for x in range(kernel_size[0]):
        for y in range(kernel_size[1]):
            x_shifted = x - center[0]
            y_shifted = y - center[1]
            if theta is not None:
                kern[x, y] = G(x_shifted, y_shifted, sigma_x, sigma_y) * M_func(x_shifted, y_shifted, f, theta)
            else:
                kern[x, y] = G(x_shifted, y_shifted, sigma_x, sigma_y) * M_func(x_shifted, y_shifted, f)
    return kern


# Define a function to apply the filter to an image
# filter_image 函数用于应用一个滤波核（kernel）到一个图像上。
# 这是一个卷积操作，它将滤波核在图像上每个位置的对应像素值进行乘积求和，然后用这个结果替换中心像素的值。
# 简单来说，这个函数的作用是将定义好的滤波核应用到输入图像上，以执行如模糊、锐化、边缘检测等操作。
# 输出是经过滤波后的图像，其大小和类型与输入图像相同。
def filter_image(image, kernel):
    return cv2.filter2D(image, -1, kernel)


# Define a function to extract features from an image block
# 函数返回这两个统计量——平均值和平均绝对偏差，作为该图像块的特征
def extract_features(image_block):
    abs_image_block = np.abs(image_block)
    mean_val = np.mean(abs_image_block)
    abs_deviation = np.mean(np.abs(abs_image_block - mean_val))
    return mean_val, abs_deviation


# Define a function to create the feature vector
# 首先定义一个特征向量 feature_vector，它将用来储存所有的特征。
# 定义 block_size，表示要处理的小图像块的大小，在这个例子中是8x8像素。
# 然后，函数遍历 filtered_images，即之前使用特定滤波器（如Gabor滤波器或自定义滤波器）处理过的图像列表。
# 对每张滤波后的图像，函数通过嵌套循环将图像分割成8x8像素的小块。这里使用两个 for 循环分别按照图像的行(x)和列(y)来遍历。
# 对于每个确定的坐标 (x, y)，函数提取出对应的8x8像素块 block。
# 如果这个提取出的 block 的大小正好等于 block_size（8x8），说明这个块是有效的（没有越界），它将被用于特征提取。
# 函数调用 extract_features 函数来提取当前块的特征。extract_features 函数返回每个块的平均值和平均绝对偏差，这两个值一起描述了这个块的纹理信息。
# 提取出的特征被添加到 feature_vector 列表中。
# 最后，将 feature_vector 列表转换为一个NumPy数组并返回。这样可以方便地用于机器学习和数据处理。
def create_feature_vector(filtered_images):
    feature_vector = []
    block_size = (8, 8)

    # Iterate over the filtered images
    for filtered_image in filtered_images:
        # Divide the filtered image into blocks
        for x in range(0, filtered_image.shape[0], block_size[0]):
            for y in range(0, filtered_image.shape[1], block_size[1]):
                block = filtered_image[x:x + block_size[0], y:y + block_size[1]]
                if block.shape[0] == block_size[0] and block.shape[1] == block_size[1]:
                    # Extract features from each block
                    features = extract_features(block)
                    feature_vector.extend(features)

    return np.array(feature_vector)


# Load the image and select the ROI
img = cv2.imread('path_to_your_image.jpg', cv2.IMREAD_GRAYSCALE)
roi = img  # In your real application, select the ROI properly

# Create the filter kernels
# 定参数
sigma_x1, sigma_y1 = 3, 1.5 # 自行修改
sigma_x2, sigma_y2 = 4.5, 1.5   # 自行修改
frequency = 0.1  # Example frequency # 自行修改
theta = 0  # Example orientation for Gabor filter # 自行修改好像是角度 rotation 不太改
kernel_size = (31, 31)  # Example kernel size 大了捕捉更多细节，但是难跑，如果要细一点的细粒度，就小一点

import numpy as np
import cv2


# Assuming the rest of the necessary functions and imports have been defined earlier
# 大汇总

def extract_iris_features(roi, frequency, sigma_x1, sigma_y1, sigma_x2, sigma_y2, theta, kernel_size):
    """
    Extracts iris features by applying defined and Gabor filters to the region of interest (ROI),
    and creates a combined feature vector for iris recognition.

    Parameters:
    - roi: The region of interest of the iris image.
    - frequency: The frequency of the sinusoidal function used in the filters.
    - sigma_x1, sigma_y1: The space constants for the Gaussian envelope along the x and y axis for M1 filter.
    - sigma_x2, sigma_y2: The space constants for the Gaussian envelope along the x and y axis for M2 filter.
    - theta: The orientation for the Gabor filter.
    - kernel_size: The size of the filter kernels.

    Returns:
    - final_feature_vector: The combined feature vector from both filters.
    """
    # Create filter kernels for M1 and M2 functions
    m1_kernel = create_filter_kernel(M1, frequency, sigma_x1, sigma_y1, kernel_size=kernel_size)
    m2_kernel = create_filter_kernel(M2, frequency, sigma_x2, sigma_y2, theta, kernel_size=kernel_size)

    # Apply the filters to the ROI
    g1_filtered = filter_image(roi, m1_kernel)
    g2_filtered = filter_image(roi, m2_kernel)

    # Create the feature vectors for each filtered image
    feature_vector_1 = create_feature_vector([g1_filtered])
    feature_vector_2 = create_feature_vector([g2_filtered])

    # Combine feature vectors to create the final feature vector
    final_feature_vector = np.concatenate((feature_vector_1, feature_vector_2))

    return final_feature_vector


# Example usage:
final_features = extract_iris_features(roi, frequency, sigma_x1, sigma_y1, sigma_x2, sigma_y2, theta, kernel_size)
print(final_features)

'''
m1_kernel = create_filter_kernel(M1, frequency, sigma_x1, sigma_y1, kernel_size=kernel_size)
m2_kernel = create_filter_kernel(M2, frequency, sigma_x2, sigma_y2, theta, kernel_size=kernel_size)

# Apply the filters to the ROI
g1_filtered = filter_image(roi, m1_kernel)
g2_filtered = filter_image(roi, m2_kernel)

# Create the feature vectors
feature_vector_1 = create_feature_vector([g1_filtered])
feature_vector_2 = create_feature_vector([g2_filtered])

# Combine feature vectors if necessary
final_feature_vector = np.concatenate((feature_vector_1, feature_vector_2))

print(final_feature_vector)
'''