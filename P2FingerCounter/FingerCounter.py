import cv2 as cv
import time
import os

import HandTrackingModule as htm

##########################
wCam, hCam = 1240, 480
##########################

capture = cv.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)

# Load and store finger images
folderPath = "FingerImages"
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

tipIDs = [4, 8, 12, 16, 20]
# 4 for thumb
# 8 for index finger
# 12 for middle finger
# 16 for ring finger
# 20 for pinky finger

while True:
    success, img = capture.read()
    if success:
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        # print(lmList)

        if len(lmList) != 0:
            fingers = []
            # For thumb
            # if lmList[tipIDs[0]][1] > lmList[tipIDs[0] - 1][1]: # Only works for RIGHT hand
            if lmList[tipIDs[0]][1] < lmList[tipIDs[0] - 1][1]: # Only works for LEFT hand
                fingers.append(1)
            else:
                fingers.append(0)

            # For other fingers
            for id in range(1, 5):
                # if lmList[tipIDs[id]][2] > lmList[tipIDs[id] - 2][2]: # Only works for RIGHT hand
                if lmList[tipIDs[id]][2] < lmList[tipIDs[id] - 2][2]: # Only works for LEFT hand
                    fingers.append(1)
                else:
                    fingers.append(0)

            # print(fingers)
            totalOpenFingers = fingers.count(1)
            print(totalOpenFingers)

            h, w, c = overLayList[totalOpenFingers - 1].shape
            img[0:h, 0:w] = overLayList[totalOpenFingers - 1]

            cv.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv.FILLED)
            cv.putText(img, str(totalOpenFingers), (45, 375), cv.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

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