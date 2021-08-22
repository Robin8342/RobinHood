#넓은 범위를 기준으로 할때는 작동되지만 세세한 결과값은 얻기 힘듬.

import cv2

img_color = cv2.imread('test.jpg')
height,width = img_color.shape[:2]

img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

lower_blue = (120-10,30,30)
upper_blue = (120+10,255,255)

img_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

img_result = cv2.bitwise_and(img_color,img_color,mask = img_mask)

cv2.imshow('img_color',img_color)
cv2.imshow('img_mask',img_mask)
cv2.imshow('img_result',img_result)

cv2.waitkey(0)
cv2.destroyAllWindows()