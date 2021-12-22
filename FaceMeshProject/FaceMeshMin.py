import cv2 as cv
import mediapipe as mp
import time

capture = cv.VideoCapture('Videos/05.mp4')
pTime = 0
cTime = 0

mpDraw = mp.solutions.drawing_utils

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    success, img = capture.read()
    if success:
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # print(id, x, y)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv.putText(img, f'FPS:{int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        cv.imshow("Result", img)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
cv.destroyAllWindows()