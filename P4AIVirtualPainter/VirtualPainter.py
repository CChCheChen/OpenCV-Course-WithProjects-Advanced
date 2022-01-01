import cv2 as cv
import numpy as np
import time
import os

import HandTrackingModule as htm

##########################
wCam, hCam = 1280, 720
brushThickness = 15
eraserThickness = 50
##########################

capture = cv.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)

# Load and store finger images
folderPath = "Headers"
myList = os.listdir(folderPath)
# print(myList)
overLayList = []
for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overLayList.append(image)
# print(len(overLayList))
header = overLayList[3]
drawColor = (0, 0, 255)

detector = htm.handDetector(detectionConf=0.85)

xp, yp = 0, 0

imgCanvas = np.zeros((hCam, wCam, 3), np.uint8)

while True:
    # 1. Import image
    success, img = capture.read()
    if success:
        # 1a. Flip the image
        img = cv.flip(img, 1)

        # 2. Find hand landmarks
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            # print(lmList)
            # Find tips of index finger and middle finger
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            # 3. Check which finger(s) is up or not. Draw when index finger is up; Select color when 2 fingers are up.
            fingers = detector.fingersUp()
            # print(fingers)

            # 4. If 2 fingers are up (in Selection mode)
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                # print("Selection Mode")
                # Checking for the click
                if y1 < 125:
                    if 140 < x1 < 270: # red
                        header = overLayList[3]
                        drawColor = (0, 0, 255)
                    elif 430 < x1 < 560: # blue
                        header = overLayList[0]
                        drawColor = (255, 0, 0)
                    elif 720 < x1 < 850: # green
                        header = overLayList[2]
                        drawColor = (0, 255, 0)
                    elif 1030 < x1 < 1170: # eraser
                        header = overLayList[1]
                        drawColor = (0, 0, 0)
                # cv.rectangle(img, (x1, y1 - 15), (x2, y2 + 15), drawColor, cv.FILLED)

            # 5. If 1 finger (index finger) is up (in Draw mode)
            if fingers[0]==False and fingers[1] and fingers[2]==False and fingers[3]==False and fingers[4]==False:
                # print("Draw Mode")
                cv.circle(img, (x1, y1), 15, drawColor, cv.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                if drawColor == (0, 0, 0):
                    cv.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                else:
                    cv.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                xp, yp = x1, y1

        if len(lmList) == 0:
            xp, yp = 0, 0

        imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
        _, imgInverse = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
        imgInverse = cv.cvtColor(imgInverse, cv.COLOR_GRAY2RGB)
        img = cv.bitwise_and(img, imgInverse)
        img = cv.bitwise_or(img, imgCanvas)

        # Setting the header image
        img[0:125, 0:1280] = header
        # img = cv.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

        cv.imshow("Img", img)
        # cv.imshow("ImgCanvas", imgCanvas)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
cv.destroyAllWindows()