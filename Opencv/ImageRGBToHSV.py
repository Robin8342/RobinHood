"""
image = pixel data (Blue,Green,Red) ==BGR
ex) Red = (0,0,255)
    Blue = (255,0,0)
    Yellow = (0,255,255)

image -> x,y 좌표로 color value 를 설정한다.

RGB888 -> 1 pixel 8 bit color (8*3 = 24 bit) (x,y,z)

HSV -> COLOR가 일정한 범위로 되어있어서 구하기 쉬움
    -> OPENCV에선 0~179도까지 범위를 가짐
    -> 각 범위바다 컬러값이 설정 되어 있음.

BGR to HSV
"""

import numpy as np
import cv2

color = [255,0,0]
pixel = np.uint8([[color]])

hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
hsv = hsv[0][0]

print("bgr", color)
print("hsv", hsv)


