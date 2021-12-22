import cv2 as cv
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, minDetectionConf=False, minTrackConf=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionConf = minDetectionConf
        self.minTrackConf = minTrackConf

        self.mpDraw = mp.solutions.drawing_utils

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.minDetectionConf, self.minTrackConf)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)
                    face = []
                    for id, lm in enumerate(faceLms.landmark):
                        ih, iw, ic = img.shape
                        x, y = int(lm.x * iw), int(lm.y * ih)
                        # cv.putText(img, str(id), (x, y), cv.FONT_HERSHEY_PLAIN, 0.6, (0, 255, 0), 1)
                        # print(id, x, y)
                        face.append([x, y])
                    faces.append(face)
        return img, faces

def main():
    capture = cv.VideoCapture('Videos/03.mp4')
    pTime = 0
    cTime = 0

    detector = FaceMeshDetector()

    while True:
        success, img = capture.read()
        if success:
            img, faces = detector.findFaceMesh(img)
            if len(faces) != 0:
                print(len(faces))
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