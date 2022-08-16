import cv2

i = cv2.imread('Figaro_1k/test/images/45.jpg')
new_i = cv2.resize(i, (360, 480))

print('p')