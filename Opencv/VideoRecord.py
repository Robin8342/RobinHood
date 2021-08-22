import cv2

VideoCapture = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('output.avi',fourcc,30.0,(640,480))

"""
VideoStart = cv2.VideoCapture('output.avi')

"""

while True():
    ret,img_color = VideoCapture.read()

    if ret ==False:
        continue
    """    
    if ret ==False:
        break
    """

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Color",img_color)
    cv2.imshow("Gray",img_gray)

    writer.write(img_color)
    if cv2.waitKey(1)&0xFF == 27:
        break

VideoCapture.release()
writer.release()

cv2.destroyAllWindows()


