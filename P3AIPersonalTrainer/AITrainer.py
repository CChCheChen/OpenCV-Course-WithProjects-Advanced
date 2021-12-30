import cv2 as cv
import numpy as np
import time
import os

import PoseEstimationModule as psm

##########################
wCam, hCam = 1240, 720
##########################

capture = cv.VideoCapture("Videos/3.mp4")

detector = psm.poseDetector()

count = 0
dir = 0

pTime = 0
cTime = 0

while True:
    success, img = capture.read()
    if success:
        img = cv.resize(img, (wCam, hCam))
        # img = cv.imread("Videos/ref.jpg")
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        # print(lmList)

        if len(lmList) != 0:
            # # Right arm
            # angle = detector.findAngle(img, 12, 14, 16, True)
            # Left arm
            angle = detector.findAngle(img, 11, 13, 15, True)

            # Base on observation from video, the angle is in range from 50 degree to 160 degree
            per = np.interp(angle, (190, 300), (0, 100))
            # print(angle, per)

            bar = np.interp(angle, (190, 300), (450, 100))

            # Check for the push-up
            color = (255, 0, 255)
            if per >= 95:
                color = (0, 255, 0)
                if dir == 0:
                    count += 0.5
                    dir = 1
            if per <= 10:
                color = (0, 255, 0)
                if dir == 1:
                    count += 0.5
                    dir = 0
            # print(count)
            cv.putText(img, str(int(count)), (50, 100), cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

            cv.rectangle(img, (1100, 100), (1175, 450), color, 2)
            cv.rectangle(img, (1100, int(bar)), (1175, 450), color, cv.FILLED)
            cv.putText(img, f'{int(per)}%', (1100, 500), cv.FONT_HERSHEY_PLAIN, 3, color, 2)

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