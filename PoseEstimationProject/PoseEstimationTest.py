import cv2 as cv
import time
import PoseEstimationModule as pem

capture = cv.VideoCapture('PoseVideos/03.mp4')
pTime = 0
cTime = 0
detector = pem.poseDetector()
while True:
    success, img = capture.read()
    if success:
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList[14])
            cv.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        cv.imshow("Result", img)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
cv.destroyAllWindows()