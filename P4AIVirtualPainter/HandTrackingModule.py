import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionConf = detectionConf
        self.trackConf = trackConf
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionConf, self.trackConf) # modify the parameter if needed
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIDs = [4, 8, 12, 16, 20]
        # 4 for thumb
        # 8 for index finger
        # 12 for middle finger
        # 16 for ring finger
        # 20 for pinky finger

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # hands object only uses RGB images
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNumb=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            thisHand = self.results.multi_hand_landmarks[handNumb]
            for id, lm in enumerate(thisHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # position of center
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 7, (255, 0, 0), cv.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []
        # For thumb
        # if lmList[tipIDs[0]][1] < lmList[tipIDs[0] - 1][1]: # Only works for RIGHT hand when image flipped
        if self.lmList[self.tipIDs[0]][1] > self.lmList[self.tipIDs[0] - 1][1]:  # Only works for LEFT hand when image flipped
            fingers.append(1)
        else:
            fingers.append(0)

        # For other fingers
        for id in range(1, 5):
            # if lmList[tipIDs[id]][2] > lmList[tipIDs[id] - 2][2]: # Only works for RIGHT hand
            if self.lmList[self.tipIDs[id]][2] < self.lmList[self.tipIDs[id] - 2][2]:  # Only works for LEFT hand
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def main():
    pTime = 0
    cTime = 0
    capture = cv.VideoCapture(0)
    detector = handDetector()

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

if __name__ == "__main__":
    main()