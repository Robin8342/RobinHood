import cv2 as cv

CAM_ID = 0

#그냥 사진 바로 찍힘.
def capture(camid = CAM_ID):
    cam = cv.VideoCapture(camid, cv.CAP_DSHOW)

    if cam.isOpened() == False:
        print("cant open the cam (%d)" % camid)
        return None

    ret, frame = cam.read()
    if frame is None:
        print('frame is not exist')
        return None

    cv.imwrite("testimage.png", frame, params=[cv.IMWRITE_PNG_COMPRESSION,0])
    cam.release()

if __name__ == '__main__':
    capture()
