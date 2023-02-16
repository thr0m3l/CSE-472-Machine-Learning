import cv2
import numpy as np
import glob
import os

data_dir = os.path.join('./NumtaDB_with_aug', '')
paths_train_a = sorted(glob.glob(os.path.join(data_dir, 'training-a', '*.png')))

random_image = paths_train_a[0]
print(random_image)

# Load the image
img = cv2.imread(random_image)

#Increase the contrast
# img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
# Convert to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert the image to binary
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Invert the image if background is white
if np.mean(img) > 128:
    img = 255 - img

# Define the structuring element for img
kernel = np.ones((2,2), np.uint8)

# Perform the img operation
img = cv2.dilate(img, kernel, iterations=1)
img = cv2.erode(img, kernel, iterations=1)

img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Apply cropping by dropping rows and columns with average pixel intensity >= 253
# (i.e. the background)
# Iterate over rows and columns and drop them if they have average pixel intensity >= 253




#Show the image
# cv2.imshow("Dilated Image", img)

#save the image
cv2.imwrite("dilated_image.png", img)