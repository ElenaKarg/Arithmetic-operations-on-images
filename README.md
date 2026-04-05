# Arithmetic-operations-on-images
Perform addition, substraction, multiplication and division

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread('C:/Users/elena/Desktop/mushroom.jpg')
img2 = cv2.imread('C:/Users/elena/Desktop/butterfly.jpg')

# Resize second image to match first
img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Convert to float for safe operations
img1_f = img1.astype(np.float32)
img2_f = img2_resized.astype(np.float32)

# Perform operations
added = cv2.add(img1_f, img2_f)          # pixel-wise addition
subtracted = cv2.subtract(img1_f, img2_f) # pixel-wise subtraction
multiplied = cv2.multiply(img1_f, img2_f / 255.0)  # normalize second image for safe multiplication
divided = cv2.divide(img1_f, img2_f + 1)  # add 1 to avoid division by zero

# Clip results to valid 0-255 and convert back to uint8
added = np.clip(added, 0, 255).astype(np.uint8)
subtracted = np.clip(subtracted, 0, 255).astype(np.uint8)
multiplied = np.clip(multiplied, 0, 255).astype(np.uint8)
divided = np.clip(divided, 0, 255).astype(np.uint8)

# Convert BGR to RGB for visualization
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2RGB)
added_rgb = cv2.cvtColor(added, cv2.COLOR_BGR2RGB)
subtracted_rgb = cv2.cvtColor(subtracted, cv2.COLOR_BGR2RGB)
multiplied_rgb = cv2.cvtColor(multiplied, cv2.COLOR_BGR2RGB)
divided_rgb = cv2.cvtColor(divided, cv2.COLOR_BGR2RGB)

# Visualize
plt.figure(figsize=(15, 10))

titles = ['Original 1', 'Original 2', 'Addition', 'Subtraction', 'Multiplication', 'Division']
images = [img1_rgb, img2_rgb, added_rgb, subtracted_rgb, multiplied_rgb, divided_rgb]

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
