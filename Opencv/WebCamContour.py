#특정 영역의 경계를 따라 같은 픽셀 강도를 따라 연결하는 방식, 모양 분석이나 오브젝트 디텍팅에 사용된다.
#contours, hierarchy = cv.findContours(image, mode)  -> image는 입력 이미지인데 바이너리 이미지여야 된다.
#contours은 리스트로 저장되며 오브젝트의 외곽선들을 저장한다.
#mode -> 검출된 정보를 리스트로 저장하는 방식을 지정한다.
#cv.drawContours(image, contours, contourldx, color) 이미지는 컬러. conturs의 모양, contourldx conturs모양 지정


import cv2 as cv

img_color = cv.imread('test.png')
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
ret, img_binary = cv.threshold(img_gray, 127,255,0)


#cv.CHAIN_APPROX_SIMPLE : 이미지의 각 꼭짓점에 표시를 나타낸다.
#CV.CHAIN_APPROX_NONE : 이미지의 각 꼭짓점을 이어서 선으로 나타낸다.
#contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

""" 삼각형이나 사각형의 색상을 서로 다르게 측정한다. 
cv.drawContours(img_color, contours, 0, (0,255,0), 3)
cv.drawContours(img_color, contours, 1, (255,0,0), 3)
"""

#finContours(CHAIN_APPROX_NON 과 CHAIN_APPROX_SIMPLE을 나타내는 코드
"""
for cnt in contours:
    for p in cnt:
        cv.circle(img_color, (p[0][0], p[0][1]), 10, (255,0,0),-1) #-1은 파란색
"""

#hierarchy 출력 값, 즉 사각형 안의 사각형이 있다면 다중으로 디텍팅 해준다.
for cnt in contours:
    cv.drawContours(img_color, [cnt],0,(255,0,0),3)

print(hierarchy)




#Contours 검출 함수 -> cv.findContours(imge, mode, method) : 해당 contour를 포인터로 찍어서 나타낸다.
#Contours 는 영역 크기, 근사화, 무게중심, 경계 사각형, Convex Hull, Convexity Defects
#근사화 : 뾰죡뾰족한 톱니 바퀴더라도 사각형으로 근사화 시킨다.
#Convex Hull : 손가락의 사이사이를 표시한다.
#Convexity Defects : 손가락의 끝부분을 이어서 디텍팅 해준다.


cv.imshow("result",img_color)
cv.waitKey(0)

