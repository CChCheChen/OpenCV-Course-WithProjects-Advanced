import cv2 as cv
import time
import HandTrackingModule as htm

pTime = 0
cTime = 0
capture = cv.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = capture.read()
    if success:
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

        cv.imshow("Result", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
cv.destroyAllWindows()