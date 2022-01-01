import cv2 as cv
import numpy as np
import time
import autopy as ap

import HandTrackingModule as htm

##########################
wCam, hCam = 1280, 720
frameReduction = 100
smoothen = 7
##########################

capture = cv.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)

detector = htm.handDetector(maxHands=1)

wScreen, hScreen = ap.screen.size()
# print(wScreen, hScreen)

pTime = 0
cTime = 0

pLocX, pLocY = 0, 0
cLocX, cLocY = 0, 0

while True:
    success, img = capture.read()
    if success:
        # 0. Flip the image
        img = cv.flip(img, 1)

        # 1. Find hand landmarks
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            # print(lmList)
            # 2. Find tips of index finger and middle finger
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            # print(x1, y1, x2, y2)

            # 3. Check which finger(s) is up or not. Move mode when only index finger is up; Click mode when 2 fingers are up.
            fingers = detector.fingersUp()
            # print(fingers)

            cv.rectangle(img, (frameReduction, frameReduction), (wCam - frameReduction, hCam - frameReduction),
                         (255, 0, 255), 2)

            # 4. If only 1 finger (index finger) is up (in Move mode)
            if fingers[1] == 1 and fingers[2] == 0:
                # 4a. Convert Coordinates

                x3 = np.interp(x1, (frameReduction, wCam - frameReduction), (0, wScreen))
                y3 = np.interp(y1, (frameReduction, hCam - frameReduction), (0, hScreen))
                # 4b. Smoothen values
                cLocX = pLocX + (x3 - pLocX) / smoothen
                cLocY = pLocY + (y3 - pLocY) / smoothen
                # 4c. Move mouse
                ap.mouse.move(cLocX, cLocY)
                cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)

                pLocX, pLocY = cLocX, cLocY

            # 5. If 2 fingers are up (in Click mode)
            if fingers[1] == 1 and fingers[2] == 1:
                # 5a. Find distance between 2 fingers
                length, img, lineInfo = detector.findDistance(8, 12, img)
                print(length)
                # 5b. Click mouse if distance is short
                if length < 50:
                    cv.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv.FILLED)
                    ap.mouse.click()

        # 7. Frame rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, f'FPS:{int(fps)}', (1000, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        # 8. Display
        cv.imshow("Img", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
cv.destroyAllWindows()