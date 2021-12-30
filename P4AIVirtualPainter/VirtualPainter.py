import cv2 as cv
import numpy as np
import time
import os

import  HandTrackingModule as htm

##########################
wCam, hCam = 1240, 480
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

print(len(overLayList))

pTime = 0
cTime = 0

detector = htm.handDetector(detectionConf=0.75)

while True:
    success, img = capture.read()
    if success:
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        # print(lmList)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv.putText(img, f'FPS:{int(fps)}', (1000, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        cv.imshow("Result", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
cv.destroyAllWindows()