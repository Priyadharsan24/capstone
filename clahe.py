import cv2
import numpy as np

img = cv2.imread(r'../thermal_image.jpg', cv2.IMREAD_GRAYSCALE)  # read image

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # create clahe
enhanced = clahe.apply(img)  # apply clahe

cv2.imshow('Original', img)
cv2.imshow('Enhanced', enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(r'../thermal_clahe.jpg', enhanced)  # save result
