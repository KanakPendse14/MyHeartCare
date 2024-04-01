# # preprocess.py

# import cv2
# import numpy as np

# def preprocess_image(image_path):
#     # Load the image using OpenCV
#     image = cv2.imread(image_path)
#     # Resize the image to the required input size (e.g., 224x224 for VGG16)
#     image = cv2.resize(image, (224, 224))
#     # Normalize pixel values to be in the range [0, 1]
#     image = image.astype(np.float32) / 255.0
#     # Expand dimensions to create a batch of size 1
#     image = np.expand_dims(image, axis=0)
#     return image


# import cv2
# import numpy as np

# def preprocess_image(image_path):
#     # Load the image using OpenCV
#     image = cv2.imread(image_path)
#     # Resize the image to the required input size (e.g., 224x224 for VGG16)
#     image = cv2.resize(image, (224, 224))
#     # Convert the image to RGB format (if it's in BGR format)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # Normalize pixel values to be in the range [0, 1]
#     image = image.astype(np.float32) / 255.0
#     # Add batch dimension to match the model's input shape
#     image = np.expand_dims(image, axis=0)
#     return image


# import cv2
# import numpy as np

# def preprocess_image(filepath, target_size=(64, 64)):
#     # Load the image using OpenCV
#     image = cv2.imread(filepath)
#     # Resize the image to the specified target size
#     image = cv2.resize(image, target_size)
#     # Convert the image to an array
#     x = np.array(image)
#     # Expand the dimensions of the image
#     x = np.expand_dims(x, axis=0)
#     return x
# import cv2
# import numpy as np

# def preprocess_image(filepath, target_size=(64, 64)):
#     # Load the image using OpenCV
#     image = cv2.imread(filepath)
#     # Resize the image to the specified target size
#     image = cv2.resize(image, target_size)
#     # Convert the image to grayscale
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Reshape the image to have a single channel
#     x = np.expand_dims(image_gray, axis=-1)
#     # Expand the dimensions of the image to match the input shape expected by the model
#     x = np.expand_dims(x, axis=0)
#     return x


import cv2
import numpy as np

def preprocess_image(filepath, target_size=(64, 64)):
    # Load the image using OpenCV
    image = cv2.imread(filepath)
    # Resize the image to the specified target size
    image_resized = cv2.resize(image, target_size)
    # Convert the image to float32 and rescale it to [0, 1]
    x = image_resized.astype(np.float32) / 255.0
    # Expand the dimensions of the image to match the input shape expected by the model
    x = np.expand_dims(x, axis=0)
    return x
