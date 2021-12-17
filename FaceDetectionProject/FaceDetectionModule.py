import cv2 as cv
import mediapipe as mp
import time

class faceDetector():
    def __init__(self, mindetectionConf=0.5):
        self.mindetectionConf = mindetectionConf

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.mindetectionConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results.detections)

        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return img, bboxs

    def fancyDraw(self, img, bbox, length=30, thickness=5):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h
        cv.rectangle(img, bbox, (255, 0, 255), 1)
        # Top left x and y
        cv.line(img, (x, y), (x + length, y), (255, 0, 255), thickness)
        cv.line(img, (x, y), (x, y + length), (255, 0, 255), thickness)
        # Top right x1 and y
        cv.line(img, (x1, y), (x1 - length, y), (255, 0, 255), thickness)
        cv.line(img, (x1, y), (x1, y + length), (255, 0, 255), thickness)
        # Bottom left x and y1
        cv.line(img, (x, y1), (x + length, y1), (255, 0, 255), thickness)
        cv.line(img, (x, y1), (x, y1 - length), (255, 0, 255), thickness)
        # Bottom right x1 and y1
        cv.line(img, (x1, y1), (x1 - length, y1), (255, 0, 255), thickness)
        cv.line(img, (x1, y1), (x1, y1 - length), (255, 0, 255), thickness)
        return img

def main():
    capture = cv.VideoCapture('Videos/03.mp4')
    pTime = 0
    cTime = 0

    detector = faceDetector()

    while True:
        success, img = capture.read()
        if success:
            img, bboxs = detector.findFaces(img)
            print(bboxs)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv.putText(img, f'FPS:{int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

            cv.imshow("Result", img)

            if cv.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break

    capture.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()