from imutils import contours
import numpy as np
import argparse
import cv2 as cv
import imutils

#CNN을 이용해서 카드의 실제 이미지를 수집하고 학습시켜 글자를 손쉽게 디텍팅 할 수 도 있다.

CAM_ID = 0
def capture(camid = CAM_ID):
    cam = cv.VideoCapture(camid, )


ap = argparse.ArgumentParser()


"""사진을 코드상 입력해서 디텍팅 하는 방법
#OCR-A 글꼴에 0~9의 숫자가 포함된다.
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-r", "--reference", required=True,
                help="path to reference OCR-A image")
"""


args = vars(ap.parse_args())

FIRST_NUMBER = {
    "3" : "American Express",
    "4" : "Visa",
    "5" : "MasterCard",
    "6" : "Discover Card"
}

ref = cv.imread(args["reference"])
ref = cv.cvtColor(ref, cv.COLOR_BGR2GRAY)
ref = cv.threshold(ref, 10, 255, cv.THRESH_BINARY_INV)[1]

refCnts = cv.findContours(ref.copy(), cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)
refCnts = imutils.grab_contours()(refCnts)
refCnts = contours.sort_contours(refCnts, method = "left-to-right")[0]
digits = {}

#사각형의 너비/높이의 좌표를 저장한다.
for (i, c) in enumerate(refCnts):
    (x, y, w, h) = cv.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    #ROI 크기를 57X88 픽셀 크기로 조정한다.
    roi = cv.resize(roi, (57,88))

    #0~9를 각 ROI 이미지에 연결한다.
    digits[i] = roi


#IMAGE에서 16자리 신용카드 번호를 분리한다.
#두 개의 커널을 만들어서 각각 다르게 디텍팅 한다.
rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (9,3))
sqKernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))

#이미지를 호출해서 그레이스케일화해서 사용한다.
image = cv.imread(args["image"])
image = imutils.resize(image, width=300)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#rectkernel과 gray 이미지를 이용해서 TOPHAT연산한다.
#어두운 배경에서 밝은(음영된 글자 : 신용카드 번호)가 나타난다.
tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, rectKernel)

#X방향에서 tophat 이미지의 gradient 값을 계산한다.
#신용카드 번호를 디텍팅하기 위한 작업
gradX = cv.Sobel(tophat, ddepth=cv.CV_32F, dx=1, dy=0,ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

#gradX로 인한 계산으로 인한 공백을 메우기 위해 사용한다.
gradX = cv.morphologyEx(gradX, cv.MORPH_CLOSE, rectKernel)
thresh = cv.threshold(gradX, 0, 255, cv.THRESH_BINARY, cv.THRESH_OTSU)[1]

thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, sqKernel)

#해당 범위의 결과값들을 cnt에 저장한다.

cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
locs = []

#위의 디텍팅 방법으로 신용카드의 정보를 잘라내기 위해 사용
#경계 직사각형을 계산해서 지정한다.
#다만 경계 값의 40~55 픽셀의 경우 어플리케이션의 환경에 따라 다르므로 테스트한다.
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv.boundingRect(c)
    ar = w / float(h)

    if ar > 2.5 and ar < 4.0:
        if (w> 40 and w<55) and (h>10 and h<20):
            locs.append((x, y, w, h))

#x의 값이 뒤바뀌면 안되므로 x의 기준으로 왼쪽에서 오른쪽으로 정렬한다.
#output에 신용카드 번호를 저장한다.
locs = sorted(locs, key=lambda x:x[0])
output = []

for (i, (gX, gY, gW, gH)) in enumerate(locs):
    groupOutput = []

    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    group = cv.threshold(group , 0, 255, cv.THRESH_BINARY, cv.THRESH_OTSU)[1]

    digitCnts = cv.findContours(group.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    digitCnts = imutils.grab_contours(digitCnts)
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

    #중첩 루프를 통해 신용카드 정보를 하나씩 디텍팅해서 저장한다.
    for c in digitCnts:
        (x, y, w, h) = cv.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv.resize(roi, (57, 88))

        scores = []

        for (digit, digitROI) in digits.items():
            result = cv.matchTemplate(roi, digitROI, cv.TM_CCOEFF)

            (_, score, _, _) = cv.minMaxLoc(result)
            scores.append(score)

        groupOutput.append(str(np.argmax(scores)))

    cv.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0,0,255), 2)
    cv.putText(image, "".join(groupOutput), (gX, gY - 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)

    output.extend(groupOutput)

#결과값을 터미널에 출력하고 이미지 표시한다. 물론 정보를 빼냈으니 가공하려면 해당 부분을 제외하면 된다.
print("Credit Crad Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv.imshow("Image", image)
cv.waitkey(0)