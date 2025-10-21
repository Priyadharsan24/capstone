import cv2
import numpy as np
import matplotlib.pyplot as plt

# read input image
image = cv2.imread(r'..\images\thermal_image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64)

# apply log transform
c = 255 / np.log(1 + np.max(image_rgb))
log_transformed = c * np.log(1 + image_rgb)
log_transformed = np.clip(log_transformed, 0, 255).astype(np.uint8)

# save enhanced image
log_transformed_bgr = cv2.cvtColor(log_transformed, cv2.COLOR_RGB2BGR)
cv2.imwrite(r'..\images\log_transformed.jpg', log_transformed_bgr)

# show both images
plt.figure(figsize=(10,5))
plt.subplot(1,2,1); plt.title('Original'); plt.imshow(image_rgb.astype(np.uint8)); plt.axis('off')
plt.subplot(1,2,2); plt.title('Log Transformed'); plt.imshow(log_transformed); plt.axis('off')
plt.show()
