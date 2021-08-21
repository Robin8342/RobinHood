import cv2

Img_color = cv2.imread('./Image/Coin.jpg',cv2.IMREAD_COLOR) #Image Read
Img_gray = cv2.cvtColor(Img_color, cv2.COLOR_BGR2GRAY)  #open image change Color : "GRAY"


""" #Image each Show
#cv2.namedWindow('Show Image')
#cv2.imshow('Show Image', img_color)    #show Image-> Program Title name, img_color -> address image open
#cv2.waitKey(0) #keybord wait


#cv2.imshow('Show Gray Image',img_gray)
#cv2.waitKey(0)
"""


#Image Resize Absolute Value
ColorImageSize = cv2.resize(Img_color,dsize=(640,480), interpolation=cv2.INTER_AREA)
GrayImageSize = cv2.resize(Img_gray,dsize=(800,800), interpolation=cv2.INTER_AREA)


"""
#Image Resize Relative Value
ColorImageSize = cv2.resize(Img_color,dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
GrayImageSize = cv2.resize(Img_gray,dsize=(0,0), fx=0.6, fy=0.6, interpolation=cv2.INTER_LINEAR)
"""

#imwrite -> ImageSave Address
while True:
    cv2.imshow('Show Color Image',ColorImageSize)
    cv2.imshow('Show Gray Image',GrayImageSize)
    if cv2.waitKey(1) == ord('q'):
        cv2.imwrite('./Image/ChangeColorImage.jpg',GrayImageSize)
        break




cv2.destroyAllWindows() #all windows cash memory destroy