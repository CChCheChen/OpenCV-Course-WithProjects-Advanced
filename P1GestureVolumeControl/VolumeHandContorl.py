import cv2 as cv
import time
import math
import numpy as np
import HandTrackingModule as htm

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

##########################
wCam, hCam = 1240, 480
##########################

capture = cv.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)

pTime = 0
cTime = 0

detector = htm.handDetector(detectionConf=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
print(volume.GetVolumeRange())
# volume.SetMasterVolumeLevel(0, None)
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPercent = 0
minVolBar = 400
maxVolBar = 150

while True:
    success, img = capture.read()
    if success:
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            # [4] for the tip of thumb and [8] for the tip of index finger
            # print(lmList[4], lmList[8])
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), 15, (255, 0, 255), cv.FILLED)
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)
            # print(length) # after observation, min = 20 and max = 320

            # Hand range is from 20 to 320
            # Volume range is from -65 to 0
            vol = np.interp(length, [20, 320], [minVol, maxVol])
            volBar = np.interp(length, [20, 320], [minVolBar, maxVolBar])
            volPercent = np.interp(length, [20, 320], [0, 100])
            # print(length, vol)

            volume.SetMasterVolumeLevel(vol, None)

            if length < 50:
                cv.circle(img, (cx, cy), 15, (0, 255, 0), cv.FILLED)

        cv.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 2)
        cv.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv.FILLED)
        cv.putText(img, f'{int(volPercent)}%', (40, 450), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv.putText(img, f'FPS:{int(fps)}', (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        cv.imshow("Result", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
cv.destroyAllWindows()