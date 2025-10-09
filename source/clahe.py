import cv2
import numpy as np

img = cv2.imread(r'../images/thermal_image.jpg')  # read color image

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

b, g, r = cv2.split(img)
b = clahe.apply(b)
g = clahe.apply(g)
r = clahe.apply(r)
enhanced = cv2.merge([b, g, r])

cv2.imshow('Original', img)
cv2.imshow('Enhanced', enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(r'../images/thermal_clahe.jpg', enhanced)
