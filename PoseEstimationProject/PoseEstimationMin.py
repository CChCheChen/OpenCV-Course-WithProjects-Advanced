import cv2 as cv
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

capture = cv.VideoCapture('PoseVideos/04.mp4')

pTime = 0
cTime = 0

while True:
    success, img = capture.read()
    if success:
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        # print(results.pose_landmarks)

        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv.circle(img, (cx, cy), 3, (255, 0, 0), cv.FILLED)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        cv.imshow("Result", img)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
cv.destroyAllWindows()