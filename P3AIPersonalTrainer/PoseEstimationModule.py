import cv2 as cv
import mediapipe as mp
import time
import math

class poseDetector():
    def __init__(self, mode=False, complexity=1, upBody=False, smooth=True, detectionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.complexity = complexity
        self.upBody = upBody
        self.smooth = smooth
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.upBody, self.smooth, self.detectionConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img


    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 3, (255, 0, 0), cv.FILLED)
        return self.lmList

    def findAngle(self, img, index1, index2, index3, draw=True):
        # Get the landmarks
        x1, y1 = self.lmList[index1][1:]
        x2, y2 = self.lmList[index2][1:]
        x3, y3 = self.lmList[index3][1:]

        # Find angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        if angle < 0:
            angle += 360
        # print(angle)

        # Draw
        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv.line(img, (x2, y2), (x3, y3), (255, 255, 255), 3)
            cv.circle(img, (x1, y1), 10, (0, 0, 255), cv.FILLED)
            cv.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv.circle(img, (x2, y2), 10, (0, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv.circle(img, (x3, y3), 10, (0, 0, 255), cv.FILLED)
            cv.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            # cv.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        return angle

def main():
    capture = cv.VideoCapture('PoseVideos/02.mp4')
    pTime = 0
    cTime = 0
    detector = poseDetector()
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

if __name__ == "__main__":
    main()