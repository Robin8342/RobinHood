import cv2

cap = cv2.VideoCapture(0)

#Output Video Frame
print('width :%d, height : %d' % (cap.get(3), cap.get(4)))

#Video Save
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('save.avi', fourcc, 25.0, (640, 480))

#Video Start
while(True):
    ret, frame = cap.read()    # Read 결과와 frame

    if(ret) :
        gray = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame_color', frame)    # 컬러 화면 출력
       #cv2.imshow('frame_gray', gray)    # Gray 화면 출력
        out.write(frame) #Video Recode
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()