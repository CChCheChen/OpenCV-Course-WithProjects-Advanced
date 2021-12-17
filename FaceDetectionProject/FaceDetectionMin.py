import cv2 as cv
import mediapipe as mp
import time

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.75)
mpDraw = mp.solutions.drawing_utils

capture = cv.VideoCapture('Videos/04.mp4')

pTime = 0
cTime = 0

while True:
    success, img = capture.read()
    if success:
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = faceDetection.process(imgRGB)
        # print(results.detections)

        if results.detections:
            for id, detection in enumerate(results.detections):
                # mpDraw.draw_detection(img, detection)  # use dots only for zoom-in face videos
                # print(id, detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv.rectangle(img, bbox, (255, 0, 255), 2)
                cv.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

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